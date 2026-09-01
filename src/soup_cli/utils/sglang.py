"""SGLang backend utilities for soup serve."""

import json
import logging
from typing import Any, Optional

# The OpenAI finish_reason vocabulary Soup emits, imported rather than
# redefined: a second copy of this set is exactly the kind of duplicated source
# of truth that #372, #392 and #424 were each caused by. utils.vllm is
# torch-free at import time, so this does not weigh down CLI startup.
from soup_cli.utils.vllm import _FINISH_REASONS

logger = logging.getLogger(__name__)


def decode_sglang_response(response: Any) -> dict:
    """Normalise whatever ``Runtime.generate`` returned into a dict.

    #76 — sglang 0.5.16's ``Runtime.generate`` ends with
    ``return json.dumps(response.json())``, i.e. a **string**. Indexing it as a
    dict raised ``TypeError: string indices must be integers`` on EVERY request,
    so `--backend sglang` started cleanly and then 500'd on every generation.

    Older sglang returned the dict directly, so both shapes are accepted rather
    than one hard assumption being swapped for another.
    """
    if isinstance(response, dict):
        return response
    if isinstance(response, (str, bytes, bytearray)):
        try:
            decoded = json.loads(response)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                "SGLang returned a response that is neither a dict nor JSON; "
                "the installed sglang may have changed its Runtime.generate "
                "contract again"
            ) from exc
        if not isinstance(decoded, dict):
            raise ValueError(
                f"SGLang returned JSON of type {type(decoded).__name__}, expected an object"
            )
        return decoded
    raise ValueError(
        f"SGLang returned an unsupported response type {type(response).__name__}"
    )


def resolve_sglang_finish_reason(meta_info: Any, max_tokens: Optional[int]) -> str:
    """Map SGLang ``meta_info`` to an OpenAI ``finish_reason`` (#360).

    Mirrors ``vllm.resolve_finish_reason``: trust the engine's own
    ``finish_reason`` when it maps to one Soup emits, otherwise derive it from
    the token count — a generation that spent the whole budget was truncated,
    not naturally stopped. The pre-fix code hardcoded ``"stop"``, so a client
    doing continue-on-length silently stopped early.

    SGLang reports ``finish_reason`` as ``{"type": "stop"|"length"|"abort"}`` in
    recent versions and as a bare string in older ones — both are accepted so a
    dict-only read cannot silently break a previously-working sglang.
    """
    info = meta_info if isinstance(meta_info, dict) else {}
    reported = info.get("finish_reason")
    if isinstance(reported, dict):
        reported = reported.get("type")
    if isinstance(reported, str) and reported in _FINISH_REASONS:
        return reported
    completion_tokens = info.get("completion_tokens")
    try:
        produced = int(completion_tokens) if completion_tokens is not None else 0
    except (TypeError, ValueError):
        produced = 0
    if max_tokens and produced >= int(max_tokens):
        return "length"
    return "stop"


def check_sglang_available() -> bool:
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
    trust_remote_code: bool = False,
):
    """Create an SGLang Runtime for serving.

    Args:
        model_path: Path to model or LoRA adapter directory.
        base_model: Base model ID (required if model_path is a LoRA adapter).
        is_adapter: Whether model_path is a LoRA adapter.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        mem_fraction_static: Fraction of GPU memory for static allocation.
        dtype: Data type for model weights.
        trust_remote_code: Execute the model repo's custom code on load.
            Default ``False`` -- the same default-deny the vLLM path takes.
            ``serve.py`` resolves the v0.36.0 gate once per invocation and
            passes the result here. This was previously an unconditional
            ``True`` at both call sites, so the SGLang backend ran a model's
            custom code whether or not the user opted in.

    Returns:
        (runtime, runtime_model_name) tuple.
    """
    import re

    import sglang as sgl

    # SSRF protection: block URL-based model paths
    for path_val in (model_path, base_model):
        if path_val and re.match(r'^https?://', path_val):
            raise ValueError(
                "model_path/base_model must be a local path or HuggingFace model ID, "
                "not a URL"
            )

    # For LoRA adapters, load the base model
    if is_adapter and base_model:
        runtime = sgl.Runtime(
            model_path=base_model,
            tp_size=tensor_parallel_size,
            mem_fraction_static=mem_fraction_static,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            lora_paths=[model_path],
        )
        runtime_model_name = base_model
    else:
        runtime = sgl.Runtime(
            model_path=model_path,
            tp_size=tensor_parallel_size,
            mem_fraction_static=mem_fraction_static,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        runtime_model_name = model_path

    return runtime, runtime_model_name


def create_sglang_app(
    runtime,
    runtime_model_name: str,
    model_name: str,
    max_tokens_default: int = 512,
    tokenizer=None,
):
    """Create a FastAPI app using SGLang runtime for inference.

    Args:
        runtime: SGLang Runtime instance.
        runtime_model_name: Model name used by SGLang.
        model_name: Display model name for API responses.
        max_tokens_default: Default max tokens for generation.
        tokenizer: HF tokenizer for the served model — its chat template is
            applied by the shared ``build_chat_prompt`` (#360). None degrades to
            the legacy role-prefixed format, same as the vLLM backend.

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

    # THE single prompt builder shared with the transformers and vLLM backends
    # (#332), so the three cannot drift again. Was a hand-rolled third copy.
    from soup_cli.utils.vllm import build_chat_prompt

    app = FastAPI(title="Soup Inference Server (SGLang)", version="1.0.0")

    # Loopback-only CORS (parity with the transformers / vLLM servers). The
    # wildcard the loopback fix never reached let any web page read this local
    # server's responses.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST", "OPTIONS"],
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

        # Apply the model's own chat template (legacy fallback when there is no
        # tokenizer / template), shared with the transformers + vLLM backends.
        prompt = build_chat_prompt(request.messages, tokenizer)

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
            response = decode_sglang_response(
                runtime.generate(prompt, sampling_params=sampling_params)
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
                        "finish_reason": resolve_sglang_finish_reason(
                            response.get("meta_info"), max_tokens
                        ),
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
            response = decode_sglang_response(
                runtime.generate(prompt, sampling_params=sampling_params)
            )
            response_text = response["text"]
            meta_info = response.get("meta_info")
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
                    "finish_reason": resolve_sglang_finish_reason(
                        meta_info, sampling_params.get("max_new_tokens")
                    ),
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return app
