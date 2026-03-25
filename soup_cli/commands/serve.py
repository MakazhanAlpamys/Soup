"""soup serve — local inference server with OpenAI-compatible API."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)

console = Console()


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
        help="Inference backend: transformers (default) or vllm",
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
    if backend not in ("transformers", "vllm"):
        console.print(
            f"[red]Unknown backend: {backend}[/]\n"
            "Supported backends: [bold]transformers[/], [bold]vllm[/]"
        )
        raise typer.Exit(1)

    # Auto-detect vLLM: if installed but not selected, show hint
    if backend == "transformers":
        from soup_cli.utils.vllm import is_vllm_available

        if is_vllm_available():
            console.print(
                "[dim]Hint: vLLM is installed. Use [bold]--backend vllm[/] "
                "for 2-4x better throughput.[/]"
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

    backend_label = "vLLM" if backend == "vllm" else "transformers"
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

    if backend == "vllm":
        app = _serve_vllm(
            model_path=model_path,
            base_model=base_model,
            is_adapter=is_adapter,
            max_tokens_default=max_tokens_default,
            tensor_parallel=tensor_parallel,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        # Transformers backend (original)
        model_obj, tokenizer = _load_model(
            model_path=str(model_path),
            base_model=base_model,
            is_adapter=is_adapter,
            device=device,
        )
        console.print("[bold green]Model loaded![/]")

        app = _create_app(
            model_obj=model_obj,
            tokenizer=tokenizer,
            device=device,
            model_name=str(model_path.name),
            max_tokens_default=max_tokens_default,
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
    """Load model and tokenizer."""
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
            dtype=torch.float16,
        )
        console.print(f"[dim]Loading LoRA adapter: {model_path}...[/]")
        model_obj = PeftModel.from_pretrained(base, model_path)
    else:
        console.print(f"[dim]Loading model: {model_path}...[/]")
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.float16,
        )

    model_obj.eval()
    return model_obj, tokenizer


def _generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
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
):
    """Create the FastAPI application with OpenAI-compatible endpoints."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field

    app = FastAPI(title="Soup Inference Server", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    # --- Endpoints ---

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "device": device}

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
                ),
                media_type="text/event-stream",
            )

        try:
            response_text, prompt_tokens, completion_tokens = _generate_response(
                model_obj, tokenizer, messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        except Exception:
            logger.exception("Generation error")
            raise HTTPException(status_code=500, detail="Internal server error")

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

    return app


def _stream_response(
    model, tokenizer, messages,
    max_tokens, temperature, top_p, model_name,
):
    """Generator that yields SSE chunks for streaming responses."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Generate full response (true token-by-token streaming requires TextIteratorStreamer)
    response_text, _, _ = _generate_response(
        model, tokenizer, messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

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
