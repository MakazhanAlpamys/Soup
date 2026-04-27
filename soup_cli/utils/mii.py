"""DeepSpeed-MII serve backend (v0.27.0).

MII (Model Implementations for Inference) provides high-throughput serving
with tensor parallelism. See https://github.com/deepspeedai/DeepSpeed-MII.

All imports are lazy so that `soup --help` stays fast.
"""

from __future__ import annotations

import sys
from typing import Any, Optional


def is_mii_available() -> bool:
    """Return True when the ``mii`` package is importable.

    Honours test-injected stubs: if ``sys.modules["mii"]`` is explicitly set
    to ``None`` (pytest's idiom for "pretend this module is absent"), the
    ``import mii`` statement below will raise ``ImportError`` without
    consulting disk.
    """
    if "mii" in sys.modules and sys.modules["mii"] is None:
        return False
    try:
        import mii  # noqa: F401
    except ImportError:
        return False
    return True


def create_mii_pipeline(
    model_path: str,
    tensor_parallel: int = 1,
    max_length: int = 4096,
    replica_num: int = 1,
) -> Any:
    """Create a DeepSpeed-MII pipeline.

    Args:
        model_path: HF model id or local path.
        tensor_parallel: TP size (must evenly divide GPU count).
        max_length: Max sequence length the pipeline will handle.
        replica_num: Number of replicas (for multi-node).

    Raises:
        ImportError: If the ``deepspeed-mii`` package is not installed.
    """
    if not is_mii_available():
        raise ImportError(
            "deepspeed-mii is not installed. "
            "Install with: pip install 'soup-cli[mii]' "
            "or pip install deepspeed-mii"
        )

    import mii

    return mii.pipeline(
        model_path,
        tensor_parallel=tensor_parallel,
        max_length=max_length,
        replica_num=replica_num,
    )


try:
    from pydantic import BaseModel as _BaseModel

    class _MiiMessage(_BaseModel):
        role: str
        content: str

    class _MiiChatRequest(_BaseModel):
        model: str = ""
        messages: list[_MiiMessage]
        max_tokens: Optional[int] = None
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False

except ImportError:
    _MiiMessage = None  # type: ignore[assignment]
    _MiiChatRequest = None  # type: ignore[assignment]


def _ensure_mii_request_models():
    if _MiiChatRequest is None:
        raise ImportError("pydantic is required for the MII server")
    return _MiiMessage, _MiiChatRequest


def build_mii_app(
    pipeline: Any, model_name: str, max_tokens_default: int = 512,
) -> Any:
    """Wrap a DeepSpeed-MII pipeline as a minimal OpenAI-compatible FastAPI
    app exposing ``/v1/chat/completions`` and ``/v1/models`` (#38, v0.33.0).

    The pipeline is held by closure so a single MII instance handles all
    requests (MII pipelines are thread-safe for concurrent generation).

    Args:
        pipeline: result of :func:`create_mii_pipeline` or any callable
            ``pipeline(prompts, max_new_tokens=...) -> [GeneratedResponse]``
            with ``.generated_text`` attributes.
        model_name: stable id surfaced in /v1/models and the response.
        max_tokens_default: default for requests that don't specify max_tokens.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    # Models are module-level (not closure) so FastAPI's introspection can
    # resolve forward refs.
    _ensure_mii_request_models()

    app = FastAPI(title=f"soup-cli MII serve [{model_name}]")
    # Loopback-only CORS — mirrors v0.30.0 transformers backend policy.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models")
    def _list_models() -> dict:
        return {
            "object": "list",
            "data": [{
                "id": model_name, "object": "model",
                "created": 0, "owned_by": "soup-cli-mii",
            }],
        }

    @app.post("/v1/chat/completions")
    def _chat(request: _MiiChatRequest) -> dict:
        import time
        import uuid

        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="streaming not supported on the MII backend yet",
            )
        if request.max_tokens is not None and (
            request.max_tokens < 1 or request.max_tokens > 16384
        ):
            raise HTTPException(
                status_code=400, detail="max_tokens must be in [1, 16384]",
            )
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        max_tokens = request.max_tokens or max_tokens_default
        try:
            responses = pipeline(
                [prompt],
                max_new_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        except Exception:
            raise HTTPException(status_code=500, detail="Internal server error")

        if not responses:
            raise HTTPException(status_code=500, detail="No response generated")
        first = responses[0]
        text = getattr(first, "generated_text", None) or str(first)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    return app
