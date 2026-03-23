"""vLLM backend utilities for soup serve."""


def is_vllm_available() -> bool:
    """Check if vLLM is installed."""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def get_vllm_version() -> str:
    """Get installed vLLM version."""
    try:
        import vllm

        return getattr(vllm, "__version__", "unknown")
    except ImportError:
        return "not installed"


def create_vllm_engine(
    model_path: str,
    base_model: str | None = None,
    is_adapter: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    dtype: str = "auto",
):
    """Create a vLLM AsyncLLMEngine for serving.

    Args:
        model_path: Path to model or LoRA adapter directory.
        base_model: Base model ID (required if model_path is a LoRA adapter).
        is_adapter: Whether model_path is a LoRA adapter.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Maximum sequence length. Auto-detected if None.
        dtype: Data type for model weights.

    Returns:
        (engine, engine_model_name) tuple.
    """
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    # For LoRA adapters, load the base model and apply LoRA at request time
    if is_adapter and base_model:
        engine_args = AsyncEngineArgs(
            model=base_model,
            enable_lora=True,
            max_lora_rank=128,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=True,
        )
        if max_model_len is not None:
            engine_args.max_model_len = max_model_len
        engine_model_name = base_model
    else:
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=True,
        )
        if max_model_len is not None:
            engine_args.max_model_len = max_model_len
        engine_model_name = model_path

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine, engine_model_name


def create_vllm_app(
    engine,
    engine_model_name: str,
    model_name: str,
    adapter_path: str | None = None,
    max_tokens_default: int = 512,
):
    """Create a FastAPI app using vLLM engine for inference.

    Args:
        engine: vLLM AsyncLLMEngine instance.
        engine_model_name: Model name used by vLLM engine.
        model_name: Display model name for API responses.
        adapter_path: Path to LoRA adapter (if using adapter).
        max_tokens_default: Default max tokens for generation.

    Returns:
        FastAPI application.
    """
    import json
    import time
    import uuid
    from typing import Optional

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field
    from vllm import SamplingParams

    app = FastAPI(title="Soup Inference Server (vLLM)", version="1.0.0")

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
        max_tokens: Optional[int] = None
        stream: bool = False

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "backend": "vllm"}

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

        # Build prompt from messages using a simple chat template
        prompt = _build_prompt(request.messages)

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=max_tokens,
        )

        # Build generate kwargs
        generate_kwargs = {}
        if adapter_path:
            from vllm.lora.request import LoRARequest

            generate_kwargs["lora_request"] = LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_path=adapter_path,
            )

        if request.stream:
            return StreamingResponse(
                _stream_vllm_response(
                    engine=engine,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    model_name=model_name,
                    generate_kwargs=generate_kwargs,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming
        try:
            results_generator = engine.generate(
                prompt, sampling_params, request_id, **generate_kwargs
            )
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if final_output is None:
                raise HTTPException(status_code=500, detail="No output generated")

            output = final_output.outputs[0]
            response_text = output.text
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(output.token_ids)

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

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

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

    async def _stream_vllm_response(
        engine,
        prompt: str,
        sampling_params,
        request_id: str,
        model_name: str,
        generate_kwargs: dict,
    ):
        """Stream SSE chunks from vLLM engine."""
        created = int(time.time())
        previous_text = ""

        results_generator = engine.generate(
            prompt, sampling_params, request_id, **generate_kwargs
        )

        async for request_output in results_generator:
            output = request_output.outputs[0]
            new_text = output.text[len(previous_text):]
            previous_text = output.text

            if new_text:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": new_text},
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
