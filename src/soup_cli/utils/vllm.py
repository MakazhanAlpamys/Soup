"""vLLM backend utilities for soup serve."""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator

logger = logging.getLogger(__name__)

# OpenAI's finish_reason vocabulary that Soup emits. Anything the engine
# reports outside this set (vLLM also uses "abort") is normalised to "stop"
# rather than leaked to clients that branch on the value.
_FINISH_REASONS = frozenset({"stop", "length"})


def _normalise_messages(messages) -> list:
    """Accept pydantic ``ChatMessage`` objects OR plain dicts.

    The vLLM app builds pydantic models; the transformers app passes dicts.
    Both must render to the same prompt, so both funnel through here.

    Dict messages are copied through WHOLE — a chat template may read keys
    beyond role/content (``tool_calls``, ``name``, multimodal content parts),
    and dropping them would silently change what the transformers backend
    renders today.
    """
    rows = []
    for msg in messages or []:
        if isinstance(msg, Mapping):
            rows.append(dict(msg))
        else:
            content = getattr(msg, "content", "")
            rows.append(
                {
                    "role": str(getattr(msg, "role", "")),
                    "content": "" if content is None else content,
                }
            )
    return rows


def _legacy_chat_prompt(messages) -> str:
    """The pre-#332 hand-rolled prompt.

    Kept ONLY as the fallback for a model that ships no chat template — which
    is exactly what the transformers backend has always done. Every other path
    must render the model's own template; feeding this to a chat-tuned model is
    what produced the Llama-3.1-8B run-on loop in #332.
    """
    parts = []
    for msg in _normalise_messages(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def build_chat_prompt(messages, tokenizer=None) -> str:
    """Render chat messages into a prompt string (#332).

    THE single prompt builder shared by ``soup serve``'s transformers and vLLM
    backends, so the two cannot drift again. Applies the model's own chat
    template when it has one, and falls back to the legacy role-prefixed
    format only when it does not (or when no tokenizer could be loaded).

    Args:
        messages: chat messages — pydantic objects or dicts.
        tokenizer: a HF tokenizer, or None when one could not be loaded.

    Returns:
        The prompt string to hand to the engine.
    """
    template = getattr(tokenizer, "chat_template", None)
    if tokenizer is not None and template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                _normalise_messages(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001 — a broken template must not 500
            logger.warning(
                "chat template failed to render; falling back to the legacy "
                "role-prefixed prompt",
                exc_info=True,
            )
    return _legacy_chat_prompt(messages)


def resolve_finish_reason(output: Any, max_tokens: Optional[int]) -> str:
    """Map a vLLM ``CompletionOutput`` to an OpenAI ``finish_reason`` (#333).

    Trusts the engine's own ``finish_reason`` when it is one Soup emits, and
    otherwise derives it from the token count — an output that used the whole
    budget was truncated, not naturally stopped. The pre-fix code hardcoded
    ``"stop"``, so a client doing continue-on-length silently stopped early.
    """
    reported = getattr(output, "finish_reason", None)
    if isinstance(reported, str):
        if reported in _FINISH_REASONS:
            return reported
        if reported:
            # e.g. vLLM's "abort" — not an OpenAI value; fall through to the
            # token-count derivation rather than leaking it.
            logger.debug("unmapped vLLM finish_reason %r", reported)
    token_ids = getattr(output, "token_ids", None) or []
    try:
        produced = len(token_ids)
    except TypeError:  # pragma: no cover — defensive
        produced = 0
    if max_tokens and produced >= int(max_tokens):
        return "length"
    return "stop"


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
    base_model: Optional[str] = None,
    is_adapter: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    dtype: str = "auto",
    speculative_model: Optional[str] = None,
    num_speculative_tokens: int = 5,
    enable_prefix_caching: bool = False,
    quantization: Optional[str] = None,
    sleep_mode: bool = False,
    trust_remote_code: bool = False,
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
        enable_prefix_caching: Enable vLLM's automatic prefix cache — big
            win for RAG / agent workloads with shared system prompts.
        sleep_mode: Enable vLLM sleep/standby support (v0.71.21 #124 —
            requires vLLM >= 0.7; raises a friendly RuntimeError otherwise).
        trust_remote_code: Execute custom repo code on model load. Default
            False — the caller (``serve``) resolves this through the shared
            v0.36.0 ``resolve_trust_remote_code`` gate (flag + warning panel)
            so an untrusted HF repo never runs code silently.

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
            trust_remote_code=trust_remote_code,
            enable_prefix_caching=enable_prefix_caching,
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
            trust_remote_code=trust_remote_code,
            enable_prefix_caching=enable_prefix_caching,
        )
        if max_model_len is not None:
            engine_args.max_model_len = max_model_len
        engine_model_name = model_path

    # v0.35.0 #61 — Auto-quant live picker forwards the chosen quant here.
    # vLLM accepts ``quantization`` on AsyncEngineArgs (string: awq/gptq/fp8).
    # ``None`` is the default (baseline / model's native dtype).
    if quantization:
        if quantization not in ("awq", "gptq", "fp8"):
            raise ValueError(
                f"quantization must be one of awq/gptq/fp8 or None, "
                f"got {quantization!r}"
            )
        engine_args.quantization = quantization

    # Speculative decoding — use a smaller draft model for faster inference
    if speculative_model:
        import re
        # SSRF protection: block URL-based model paths
        if re.match(r'^https?://', speculative_model):
            raise ValueError(
                "speculative_model must be a local path or HuggingFace model ID, "
                "not a URL"
            )
        engine_args.speculative_model = speculative_model
        engine_args.num_speculative_tokens = num_speculative_tokens

    # v0.71.21 #124 — vLLM sleep/standby support (engine.sleep()/wake_up()).
    # apply_vllm_sleep_mode raises a friendly RuntimeError on vLLM < 0.7.
    if sleep_mode:
        from soup_cli.utils.grpo_long_context import apply_vllm_sleep_mode

        apply_vllm_sleep_mode(engine_args)

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine, engine_model_name


def create_vllm_app(
    engine,
    engine_model_name: str,
    model_name: str,
    adapter_path: Optional[str] = None,
    max_tokens_default: int = 512,
    tokenizer=None,
    enable_dashboard: bool = False,
):
    """Create a FastAPI app using vLLM engine for inference.

    Args:
        engine: vLLM AsyncLLMEngine instance.
        engine_model_name: Model name used by vLLM engine.
        model_name: Display model name for API responses.
        adapter_path: Path to LoRA adapter (if using adapter).
        max_tokens_default: Default max tokens for generation.
        tokenizer: HF tokenizer for the served model — its chat template is
            what turns messages into the prompt (#332). ``None`` degrades to
            the legacy role-prefixed format.
        enable_dashboard: Recorded on ``app.state`` for introspection.
            ``/metrics`` is served either way, mirroring the transformers
            backend (#333 — the flag used to no-op entirely here).

    Returns:
        FastAPI application.
    """
    import contextlib
    import json
    import time
    import uuid

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field
    from vllm import SamplingParams

    from soup_cli.utils.metrics import ServerMetrics

    app = FastAPI(title="Soup Inference Server (vLLM)", version="1.0.0")

    # #333 — shared metrics bucket. Created unconditionally so /metrics works
    # whether or not --dashboard is set, exactly like the transformers app.
    metrics = ServerMetrics()

    # v0.53.7 M-G: restrict CORS to loopback — /v1/messages is a mutation
    # endpoint and was previously reachable from any browser page via the
    # legacy ``allow_origins=["*"]``. Matches v0.30.0 transformers-backend
    # policy.
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
        return {"status": "ok", "model": model_name, "backend": "vllm"}

    @app.get("/metrics")
    def metrics_endpoint():
        """Dashboard + Prometheus-style JSON scrape (#333)."""
        return metrics.snapshot()

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

        # #332 — the model's OWN chat template, shared with the transformers
        # backend. The pre-fix hand-rolled prompt made chat-tuned models loop.
        prompt = build_chat_prompt(request.messages, tokenizer)

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
                    max_tokens=max_tokens,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming
        started = time.perf_counter()
        with contextlib.ExitStack() as stack:
            stack.enter_context(metrics.track_request())
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
                metrics.record_tokens(completion_tokens)

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
                            # #333 — was hardcoded "stop", so a length
                            # truncation looked like a natural stop.
                            "finish_reason": resolve_finish_reason(
                                output, max_tokens
                            ),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }

            except HTTPException:
                raise
            except Exception:
                logger.exception("vLLM generation error")
                raise HTTPException(status_code=500, detail="Internal server error")
            finally:
                # Record latency on error paths too, or the dashboard shows a
                # blind spot exactly when things are going wrong.
                metrics.record_latency((time.perf_counter() - started) * 1000)

    async def _stream_vllm_response(
        engine,
        prompt: str,
        sampling_params,
        request_id: str,
        model_name: str,
        generate_kwargs: dict,
        max_tokens: Optional[int] = None,
    ):
        """Stream SSE chunks from vLLM engine."""
        created = int(time.time())
        previous_text = ""
        last_output = None
        started = time.perf_counter()

        with metrics.track_request():
            try:
                results_generator = engine.generate(
                    prompt, sampling_params, request_id, **generate_kwargs
                )

                async for request_output in results_generator:
                    output = request_output.outputs[0]
                    last_output = output
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

                if last_output is not None:
                    metrics.record_tokens(len(getattr(last_output, "token_ids", ()) or ()))
            finally:
                metrics.record_latency((time.perf_counter() - started) * 1000)

        # Final chunk — #333: report the real reason, not a hardcoded "stop".
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": resolve_finish_reason(last_output, max_tokens),
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    # ----- v0.53.7 #102 — Anthropic /v1/messages on vLLM backend -----
    # Mirrors the v0.53.6 transformers-backend route. Reuses the v0.45.0
    # Part B converter + the existing chat_completions handler; streaming
    # emits the Anthropic event shape.
    @app.post("/v1/messages")
    async def anthropic_messages(payload: dict):
        from soup_cli.utils.anthropic_messages import (
            from_anthropic,
            validate_anthropic_payload,
        )

        wants_stream = isinstance(payload, dict) and bool(payload.get("stream"))

        try:
            validate_anthropic_payload(payload)
            openai_payload = from_anthropic(payload)
            openai_payload.pop("stream", None)
            req = ChatCompletionRequest(**openai_payload)
        except (TypeError, ValueError) as exc:
            logger.debug("vLLM /v1/messages invalid request: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid request")
        except Exception as exc:  # noqa: BLE001
            logger.debug("vLLM /v1/messages pydantic error: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid request")

        chat_response = await chat_completions(req)

        text = ""
        if isinstance(chat_response, dict):
            try:
                text = chat_response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = ""
        usage = (
            chat_response.get("usage", {}) if isinstance(chat_response, dict) else {}
        )
        msg_id = (
            chat_response.get("id", "") if isinstance(chat_response, dict) else ""
        )
        out_model = openai_payload.get("model", model_name)
        in_tokens = int(usage.get("prompt_tokens", 0) or 0)
        out_tokens = int(usage.get("completion_tokens", 0) or 0)
        # #333 — carry the real finish reason across to Anthropic's vocabulary
        # instead of always claiming a natural end_turn.
        stop_reason = _anthropic_stop_reason(chat_response)

        if wants_stream:
            # v0.53.7 L-C: disable intermediate caching on SSE streams.
            return StreamingResponse(
                _stream_anthropic_messages_vllm(
                    msg_id=msg_id,
                    model=out_model,
                    text=text,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    stop_reason=stop_reason,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-store",
                    "X-Accel-Buffering": "no",
                },
            )

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": out_model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
            },
        }

    # Introspection parity with the transformers app.
    app.state.enable_dashboard = enable_dashboard
    app.state.has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    return app


def _anthropic_stop_reason(chat_response) -> str:
    """Map an OpenAI ``finish_reason`` to Anthropic's ``stop_reason``."""
    reason = None
    if isinstance(chat_response, Mapping):
        try:
            reason = chat_response["choices"][0]["finish_reason"]
        except (KeyError, IndexError, TypeError):
            reason = None
    return "max_tokens" if reason == "length" else "end_turn"


def _stream_anthropic_messages_vllm(
    *,
    msg_id: str,
    model: str,
    text: str,
    input_tokens: int,
    output_tokens: int,
    stop_reason: str = "end_turn",
) -> "Generator[str, None, None]":
    """Yield Anthropic event-shape SSE frames (vLLM backend)."""
    import json as _json

    # M-A: sanitise caller-influenced fields before SSE embedding.
    def _sanitise(value: str, max_len: int) -> str:
        if not isinstance(value, str):
            return ""
        return value.replace("\r", "").replace("\n", "").replace("\x00", "")[:max_len]

    msg_id = _sanitise(msg_id, 64)
    model = _sanitise(model, 200)

    def _frame(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {_json.dumps(data)}\n\n"

    yield _frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "usage": {
                    "input_tokens": int(input_tokens),
                    "output_tokens": 0,
                },
            },
        },
    )
    words = (text or "").split(" ")
    for idx, word in enumerate(words):
        chunk_text = word if idx == 0 else f" {word}"
        yield _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk_text},
            },
        )
    yield _frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": int(output_tokens)},
        },
    )
    yield _frame("message_stop", {"type": "message_stop"})
