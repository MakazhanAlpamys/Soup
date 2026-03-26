"""SGLang backend utilities for soup serve."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_sglang_available() -> bool:
    """Check if SGLang is installed."""
    try:
        import sglang  # noqa: F401

        return True
    except ImportError:
        return False


def get_sglang_version() -> str:
    """Get installed SGLang version."""
    try:
        import sglang

        return getattr(sglang, "__version__", "unknown")
    except ImportError:
        return "not installed"


def create_sglang_runtime(
    model_path: str,
    base_model: Optional[str] = None,
    is_adapter: bool = False,
    tensor_parallel_size: int = 1,
    mem_fraction_static: float = 0.88,
    dtype: str = "auto",
):
    """Create an SGLang Runtime for serving.

    Args:
        model_path: Path to model or LoRA adapter directory.
        base_model: Base model ID (required if model_path is a LoRA adapter).
        is_adapter: Whether model_path is a LoRA adapter.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        mem_fraction_static: Fraction of GPU memory for static allocation.
        dtype: Data type for model weights.

    Returns:
        (runtime, runtime_model_name) tuple.
    """
    import sglang as sgl

    # For LoRA adapters, load the base model
    if is_adapter and base_model:
        runtime = sgl.Runtime(
            model_path=base_model,
            tp_size=tensor_parallel_size,
            mem_fraction_static=mem_fraction_static,
            dtype=dtype,
            trust_remote_code=True,
            lora_paths=[model_path],
        )
        runtime_model_name = base_model
    else:
        runtime = sgl.Runtime(
            model_path=model_path,
            tp_size=tensor_parallel_size,
            mem_fraction_static=mem_fraction_static,
            dtype=dtype,
            trust_remote_code=True,
        )
        runtime_model_name = model_path

    return runtime, runtime_model_name


def create_sglang_app(
    runtime,
    runtime_model_name: str,
    model_name: str,
    max_tokens_default: int = 512,
):
    """Create a FastAPI app using SGLang runtime for inference.

    Args:
        runtime: SGLang Runtime instance.
        runtime_model_name: Model name used by SGLang.
        model_name: Display model name for API responses.
        max_tokens_default: Default max tokens for generation.

    Returns:
        FastAPI application.
    """
    import json
    import time
    import uuid

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field

    app = FastAPI(title="Soup Inference Server (SGLang)", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "backend": "sglang"}

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
    async def chat_completions(request: ChatCompletionRequest):
        max_tokens = request.max_tokens or max_tokens_default
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Build prompt from messages
        prompt = _build_prompt(request.messages)

        sampling_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": max_tokens,
        }

        if request.stream:
            return StreamingResponse(
                _stream_sglang_response(
                    runtime=runtime,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    model_name=model_name,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming
        try:
            response = runtime.generate(
                prompt,
                sampling_params=sampling_params,
            )
            response_text = response["text"]
            prompt_tokens = response.get("meta_info", {}).get("prompt_tokens", 0)
            completion_tokens = response.get(
                "meta_info", {},
            ).get("completion_tokens", len(response_text.split()))

            return {
                "id": request_id,
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

        except Exception:
            logger.exception("SGLang generation error")
            raise HTTPException(status_code=500, detail="Internal server error")

    def _build_prompt(messages: list[ChatMessage]) -> str:
        """Build a simple prompt from chat messages."""
        parts = []
        for msg in messages:
            role = msg.role
            content = msg.content
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    async def _stream_sglang_response(
        runtime,
        prompt: str,
        sampling_params: dict,
        request_id: str,
        model_name: str,
    ):
        """Stream SSE chunks from SGLang runtime."""
        created = int(time.time())

        try:
            response = runtime.generate(
                prompt,
                sampling_params=sampling_params,
            )
            response_text = response["text"]
        except Exception:
            logger.exception("SGLang stream error")
            yield 'data: {"error": "Internal server error"}\n\n'
            return

        # Simulate streaming by sending word-by-word
        words = response_text.split(" ")
        for idx, word in enumerate(words):
            chunk_text = word if idx == 0 else f" {word}"
            chunk = {
                "id": request_id,
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
            "id": request_id,
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

    return app
