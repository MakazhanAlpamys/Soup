"""soup data generate — generate synthetic training data using LLMs."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

logger = logging.getLogger(__name__)

console = Console()

VALID_PROVIDERS = ("openai", "local", "server", "ollama", "anthropic", "vllm")
VALID_FORMATS = ("alpaca", "sharegpt", "chatml")
VALID_TEMPLATES = ("code", "conversation", "qa", "preference", "reasoning")


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
        help="LLM provider: openai, local, server, ollama, anthropic, vllm",
    ),
    model_name: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="Model name (OpenAI model ID, Ollama model, or local model path)",
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
    # --- v0.20.0: Template support ---
    template: Optional[str] = typer.Option(
        None,
        "--template",
        help="Domain template: code, conversation, qa, preference, reasoning",
    ),
    template_language: str = typer.Option(
        "Python",
        "--language",
        help="Language for code template (Python, JavaScript, Go, Rust, Java)",
    ),
    template_task_type: str = typer.Option(
        "function",
        "--task-type",
        help="Task type for code template (function, debug, explain, refactor, test)",
    ),
    template_turns: int = typer.Option(
        4,
        "--turns",
        help="Number of turns for conversation template (2-10)",
    ),
    template_topic: str = typer.Option(
        "general knowledge",
        "--topic",
        help="Topic for conversation template",
    ),
    template_context: Optional[str] = typer.Option(
        None,
        "--context",
        help="Path to context document for QA template",
    ),
    template_pref_task: str = typer.Option(
        "dpo",
        "--pref-task",
        help="Preference task format: dpo, kto, orpo",
    ),
    template_domain: str = typer.Option(
        "math",
        "--domain",
        help="Domain for reasoning template (math, logic, code)",
    ),
    # --- v0.20.0: Ollama-specific ---
    ollama_model: Optional[str] = typer.Option(
        None,
        "--ollama-model",
        help="Ollama model name (e.g. llama3.1). Shorthand for --provider ollama --model X",
    ),
    # --- v0.20.0: Quality pipeline ---
    validate_output: bool = typer.Option(
        False,
        "--validate",
        help="Auto-validate generated data after generation",
    ),
    filter_output: bool = typer.Option(
        False,
        "--filter",
        help="Auto-filter generated data by quality (perplexity + coherence)",
    ),
    dedup_output: bool = typer.Option(
        False,
        "--dedup",
        help="Auto-dedup generated data (MinHash)",
    ),
    quality_pipeline: bool = typer.Option(
        False,
        "--quality-pipeline",
        help="Run full quality pipeline: validate + filter + dedup",
    ),
    requests_per_minute: int = typer.Option(
        60,
        "--requests-per-minute",
        "--rpm",
        help="Rate limit for API requests (default: 60)",
    ),
):
    """Generate synthetic training data using an LLM."""
    if fmt not in VALID_FORMATS:
        console.print(
            f"[red]Invalid format: {fmt}. Must be one of: {', '.join(VALID_FORMATS)}[/]"
        )
        raise typer.Exit(1)

    # Handle --ollama-model shorthand
    if ollama_model:
        provider = "ollama"
        model_name = ollama_model

    if provider not in VALID_PROVIDERS:
        console.print(
            f"[red]Invalid provider: {provider}. "
            f"Must be one of: {', '.join(VALID_PROVIDERS)}[/]"
        )
        raise typer.Exit(1)

    # Validate template
    if template and template not in VALID_TEMPLATES:
        console.print(
            f"[red]Invalid template: {template}. "
            f"Must be one of: {', '.join(VALID_TEMPLATES)}[/]"
        )
        raise typer.Exit(1)

    # Enable quality pipeline shortcut
    if quality_pipeline:
        validate_output = True
        filter_output = True
        dedup_output = True

    cwd = Path.cwd().resolve()

    # Load seed examples if provided
    seed_examples = []
    if seed_file:
        seed_path = Path(seed_file).resolve()
        if not _path_within_cwd(seed_path, cwd):
            console.print("[red]Seed file must be within the current directory[/]")
            raise typer.Exit(1)
        if not seed_path.exists():
            console.print(f"[red]Seed file not found: {seed_path}[/]")
            raise typer.Exit(1)
        from soup_cli.data.loader import load_raw_data

        seed_examples = load_raw_data(seed_path)
        console.print(f"[dim]Loaded {len(seed_examples)} seed examples[/]")

    # Load existing data for dedup
    existing_texts = set()
    if dedup_with:
        dedup_path = Path(dedup_with).resolve()
        if not _path_within_cwd(dedup_path, cwd):
            console.print("[red]Dedup file must be within the current directory[/]")
            raise typer.Exit(1)
        if not dedup_path.exists():
            console.print(f"[red]Dedup file not found: {dedup_path}[/]")
            raise typer.Exit(1)
        from soup_cli.data.loader import load_raw_data

        existing_data = load_raw_data(dedup_path)
        for row in existing_data:
            existing_texts.add(_row_to_text(row))
        console.print(f"[dim]Loaded {len(existing_texts)} existing examples for dedup[/]")

    # Load template context if provided
    context_text = ""
    if template_context:
        ctx_path = Path(template_context).resolve()
        if not _path_within_cwd(ctx_path, cwd):
            console.print("[red]Context file must be within the current directory[/]")
            raise typer.Exit(1)
        if not ctx_path.exists():
            console.print(f"[red]Context file not found: {ctx_path}[/]")
            raise typer.Exit(1)
        context_text = ctx_path.read_text(encoding="utf-8")

    # Generate
    console.print(f"[dim]Generating {count} examples using {provider}/{model_name}...[/]")
    if template:
        console.print(f"[dim]Template: {template}[/]")

    all_examples = []
    duplicates = 0
    min_interval = 60.0 / max(1, requests_per_minute)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=count)

        remaining = count
        last_request_time = 0.0
        while remaining > 0:
            current_batch = min(batch_size, remaining)

            # Rate limiting
            elapsed = time.monotonic() - last_request_time
            if last_request_time > 0 and elapsed < min_interval:
                time.sleep(min_interval - elapsed)

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
                    template=template,
                    template_language=template_language,
                    template_task_type=template_task_type,
                    template_turns=template_turns,
                    template_topic=template_topic,
                    context_text=context_text,
                    template_pref_task=template_pref_task,
                    template_domain=template_domain,
                )
            except Exception as exc:
                console.print(f"[red]Generation error: {exc}[/]")
                raise typer.Exit(1)

            last_request_time = time.monotonic()

            # Validate and dedup
            for example in batch:
                if not _validate_example(example, fmt):
                    # For preference templates, validate against preference format
                    if template == "preference" and _validate_preference(example):
                        all_examples.append(example)
                        continue
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

    # Sanitize output path (prevent path traversal)
    out_path = Path(output).resolve()
    if not _path_within_cwd(out_path, cwd):
        console.print("[red]Output path must be within the current directory[/]")
        raise typer.Exit(1)

    # Write output
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    console.print(
        f"\n[green]Generated {len(all_examples)} examples[/]\n"
        f"Format:     [bold]{fmt}[/]\n"
        f"Output:     [bold]{out_path}[/]\n"
        + (f"Template:   [bold]{template}[/]\n" if template else "")
        + (f"Duplicates: [yellow]{duplicates} removed[/]\n" if duplicates > 0 else "")
    )

    # --- Quality pipeline ---
    if validate_output:
        _run_validate_pipeline(out_path, fmt)
    if filter_output:
        _run_filter_pipeline(out_path)
    if dedup_output:
        _run_dedup_pipeline(out_path)


def _run_validate_pipeline(path: Path, fmt: str) -> None:
    """Run validation on generated output, retry malformed entries."""
    console.print("[dim]Running validation pipeline...[/]")
    from soup_cli.data.loader import load_raw_data

    data = load_raw_data(path)
    valid = []
    invalid_count = 0
    for row in data:
        if _validate_example(row, fmt) or _validate_preference(row):
            valid.append(row)
        else:
            invalid_count += 1

    if invalid_count > 0:
        with open(path, "w", encoding="utf-8") as f:
            for row in valid:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        console.print(
            f"[yellow]Validation: removed {invalid_count} malformed entries, "
            f"{len(valid)} remaining[/]"
        )
    else:
        console.print(f"[green]Validation: all {len(valid)} entries valid[/]")


def _run_filter_pipeline(path: Path) -> None:
    """Run quality filter (coherence) on generated output."""
    console.print("[dim]Running quality filter pipeline...[/]")
    from soup_cli.data.loader import load_raw_data

    data = load_raw_data(path)
    if not data:
        return

    texts = [" ".join(str(v) for v in row.values() if v) for row in data]

    try:
        from soup_cli.utils.quality import compute_coherence_score

        scores = compute_coherence_score(texts)
        # Use median as threshold — remove bottom quartile
        sorted_scores = sorted(scores)
        threshold = sorted_scores[len(sorted_scores) // 4] if sorted_scores else 0.0

        kept = [row for row, score in zip(data, scores) if score >= threshold]
        removed = len(data) - len(kept)

        if removed > 0:
            with open(path, "w", encoding="utf-8") as f:
                for row in kept:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            console.print(
                f"[yellow]Filter: removed {removed} low-quality entries, "
                f"{len(kept)} remaining[/]"
            )
        else:
            console.print(f"[green]Filter: all {len(data)} entries passed[/]")
    except ImportError:
        console.print("[yellow]Filter: quality scoring not available, skipping[/]")


def _run_dedup_pipeline(path: Path) -> None:
    """Run MinHash dedup on generated output."""
    console.print("[dim]Running dedup pipeline...[/]")
    from soup_cli.data.loader import load_raw_data

    data = load_raw_data(path)
    if not data:
        return

    try:
        from datasketch import MinHash, MinHashLSH

        num_perm = 128
        lsh = MinHashLSH(threshold=0.8, num_perm=num_perm)
        minhashes = []

        for idx, row in enumerate(data):
            text = " ".join(str(v) for v in row.values() if v).lower()
            words = text.split()
            shingles = set()
            for ii in range(max(1, len(words) - 2)):
                shingles.add(" ".join(words[ii: ii + 3]))

            mhash = MinHash(num_perm=num_perm)
            for shingle in shingles:
                mhash.update(shingle.encode("utf-8"))
            minhashes.append(mhash)

            try:
                lsh.insert(str(idx), mhash)
            except ValueError:
                logger.debug("LSH insert collision at idx %d", idx)

        seen: set[int] = set()
        unique_indices = []
        for idx in range(len(data)):
            if idx in seen:
                continue
            unique_indices.append(idx)
            results = lsh.query(minhashes[idx])
            for dup_idx_str in results:
                seen.add(int(dup_idx_str))

        unique_data = [data[idx] for idx in unique_indices]
        removed = len(data) - len(unique_data)

        if removed > 0:
            with open(path, "w", encoding="utf-8") as f:
                for row in unique_data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            console.print(
                f"[yellow]Dedup: removed {removed} near-duplicates, "
                f"{len(unique_data)} remaining[/]"
            )
        else:
            console.print(f"[green]Dedup: no duplicates found in {len(data)} entries[/]")
    except ImportError:
        console.print(
            "[yellow]Dedup: datasketch not installed, skipping. "
            "Install: pip install 'soup-cli[data]'[/]"
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
    template: Optional[str] = None,
    template_language: str = "Python",
    template_task_type: str = "function",
    template_turns: int = 4,
    template_topic: str = "general knowledge",
    context_text: str = "",
    template_pref_task: str = "dpo",
    template_domain: str = "math",
) -> list[dict]:
    """Generate a batch of examples using the specified provider."""
    # Build the generation prompt — template or default
    if template:
        generation_prompt = _build_template_prompt(
            template=template,
            prompt=prompt,
            count=count,
            fmt=fmt,
            language=template_language,
            task_type=template_task_type,
            turns=template_turns,
            topic=template_topic,
            context_text=context_text,
            pref_task=template_pref_task,
            domain=template_domain,
        )
    else:
        generation_prompt = _build_generation_prompt(prompt, count, fmt, seed_examples)

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
            generation_prompt=generation_prompt if template else None,
        )
    elif provider == "local":
        return _generate_local(
            prompt=prompt,
            count=count,
            fmt=fmt,
            model_name=model_name,
            temperature=temperature,
            seed_examples=seed_examples,
            generation_prompt=generation_prompt if template else None,
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
            generation_prompt=generation_prompt if template else None,
        )
    elif provider == "ollama":
        from soup_cli.data.providers.ollama import DEFAULT_OLLAMA_BASE, generate_ollama

        base_url = api_base or DEFAULT_OLLAMA_BASE
        return generate_ollama(
            prompt=prompt,
            count=count,
            fmt=fmt,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            generation_prompt=generation_prompt,
        )
    elif provider == "anthropic":
        from soup_cli.data.providers.anthropic import generate_anthropic

        return generate_anthropic(
            prompt=prompt,
            count=count,
            fmt=fmt,
            model_name=model_name,
            temperature=temperature,
            generation_prompt=generation_prompt,
        )
    elif provider == "vllm":
        from soup_cli.data.providers.vllm import DEFAULT_VLLM_BASE, generate_vllm

        base_url = api_base or DEFAULT_VLLM_BASE
        return generate_vllm(
            prompt=prompt,
            count=count,
            fmt=fmt,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            generation_prompt=generation_prompt,
        )
    return []


def _build_template_prompt(
    template: str,
    prompt: str,
    count: int,
    fmt: str,
    language: str = "Python",
    task_type: str = "function",
    turns: int = 4,
    topic: str = "general knowledge",
    context_text: str = "",
    pref_task: str = "dpo",
    domain: str = "math",
) -> str:
    """Build generation prompt from a domain template."""
    format_spec = _get_format_spec(fmt)

    if template == "code":
        from soup_cli.data.templates.code import build_prompt

        return build_prompt(count, fmt, format_spec, language=language, task_type=task_type)
    elif template == "conversation":
        from soup_cli.data.templates.conversation import build_prompt

        return build_prompt(count, fmt, format_spec, turns=turns, topic=topic)
    elif template == "qa":
        from soup_cli.data.templates.qa import build_prompt

        return build_prompt(count, fmt, format_spec, context=context_text)
    elif template == "preference":
        from soup_cli.data.templates.preference import build_prompt

        return build_prompt(count, task=pref_task)
    elif template == "reasoning":
        from soup_cli.data.templates.reasoning import build_prompt

        return build_prompt(count, fmt, format_spec, domain=domain)

    # Fallback to default prompt
    return _build_generation_prompt(prompt, count, fmt, [])


def _get_format_spec(fmt: str) -> str:
    """Get the format specification string for the given format."""
    specs = {
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
    return specs.get(fmt, specs["alpaca"])


def _build_generation_prompt(prompt: str, count: int, fmt: str, seed_examples: list) -> str:
    """Build the prompt for data generation."""
    format_spec = _get_format_spec(fmt)

    system_msg = (
        f"You are a training data generator. Generate exactly {count} diverse, "
        f"high-quality training examples.\n\n"
        f"Topic/Instructions: {prompt}\n\n"
        f"Format: {format_spec}\n\n"
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
    generation_prompt: Optional[str] = None,
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

    if generation_prompt is None:
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
    generation_prompt: Optional[str] = None,
) -> list[dict]:
    """Generate examples using a local model via transformers."""
    import torch
    from rich.panel import Panel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(Panel(
        f"[yellow]Loading model with trust_remote_code=True: {model_name}[/]\n"
        "This executes code from the model repository.",
        title="Security Warning",
        style="yellow",
    ))

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

    if generation_prompt is None:
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
    generation_prompt: Optional[str] = None,
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

    if generation_prompt is None:
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
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected server response format: {exc}") from exc

    return _parse_json_array(content)


def _parse_json_array(content: str) -> list[dict]:
    """Parse a JSON array from LLM output, handling markdown code blocks.

    Delegates to soup_cli.data.providers._utils.parse_json_array to avoid
    circular imports (providers import this, this imports providers).
    """
    from soup_cli.data.providers._utils import parse_json_array

    return parse_json_array(content)


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


def _validate_preference(example: dict) -> bool:
    """Validate a preference data example (DPO/KTO format)."""
    # DPO/ORPO format
    if "prompt" in example and "chosen" in example and "rejected" in example:
        return True
    # KTO format
    if "prompt" in example and "completion" in example and "label" in example:
        return True
    return False


def _path_within_cwd(path: Path, cwd: Path) -> bool:
    """Check that a resolved path is within the current working directory."""
    try:
        path.relative_to(cwd)
        return True
    except ValueError:
        return False


def _row_to_text(row: dict) -> str:
    """Convert a row to a text string for dedup comparison."""
    return " ".join(str(v) for v in row.values() if v)
