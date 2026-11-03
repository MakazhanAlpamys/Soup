"""Bounded single-worker MLX serving inside the existing Soup HTTP application.

The worker owns model loading, decoding and cleanup on one thread. HTTP handlers
never touch MLX tensors. Each job has a fresh engine iterator, KV cache and seed.
"""

import asyncio
import json
import logging
import os
import queue
import secrets
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from soup_cli.utils.mlx_generation import NativeGenerationChunk, NativePromptError

logger = logging.getLogger(__name__)
MAX_BODY_BYTES = 1024 * 1024
MAX_PROMPT_TOKENS = 16384
QUEUE_CAPACITY = 8
OUTPUT_PUT_TIMEOUT_SECONDS = 30.0


class NativeMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=65536)


class NativeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    messages: list[NativeMessage] = Field(min_length=1, max_length=128)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1, le=16384)
    stream: bool = False
    stop: str | list[str] | None = None

    @field_validator("stop")
    @classmethod
    def bounded_stops(cls, value: str | list[str] | None) -> str | list[str] | None:
        items = [value] if isinstance(value, str) else value or []
        if len(items) > 4 or any(not 1 <= len(item) <= 256 for item in items):
            raise ValueError("stop accepts at most four nonempty strings of at most 256 characters")
        return value


class _Protocol(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class NativeUsage(_Protocol):
    prompt_tokens: int = Field(ge=1)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=1)


class NativeAssistantMessage(_Protocol):
    role: Literal["assistant"]
    content: str


class NativeCompletionChoice(_Protocol):
    index: Literal[0]
    message: NativeAssistantMessage
    finish_reason: Literal["stop", "length"]


class _CompletionIdentity(_Protocol):
    id: str
    created: int = Field(ge=0)
    model: str


class NativeCompletion(_CompletionIdentity):
    object: Literal["chat.completion"]
    choices: list[NativeCompletionChoice] = Field(min_length=1, max_length=1)
    usage: NativeUsage


class NativeDelta(_Protocol):
    role: Literal["assistant"] | None = None
    content: str | None = None


class NativeStreamChoice(_Protocol):
    index: Literal[0]
    delta: NativeDelta
    finish_reason: Literal["stop", "length"] | None


class NativeStreamFrame(_CompletionIdentity):
    object: Literal["chat.completion.chunk"]
    choices: list[NativeStreamChoice] = Field(min_length=1, max_length=1)
    usage: NativeUsage | None = None


class NativeHealth(_Protocol):
    status: Literal["ok"]
    model: str
    device: Literal["mlx"]


class NativeModelIdentity(_Protocol):
    id: str
    object: Literal["model"]
    owned_by: Literal["soup"]


class NativeModels(_Protocol):
    object: Literal["list"]
    data: list[NativeModelIdentity] = Field(min_length=1, max_length=1)


def validate_serve_options(options: dict[str, tuple[Any, Any]]) -> None:
    """Refuse backend-specific flags before optional imports, I/O or allocation."""
    for flag, (value, default) in options.items():
        if value != default:
            raise ValueError(f"--backend mlx does not support {flag}; omit it")


def validate_serve_paths(model: str, base: str | None) -> None:
    """Keep operator-selected local paths under cwd, with no symlink components."""
    from soup_cli.utils.mlx_generation import read_native_adapter
    from soup_cli.utils.paths import is_under_cwd

    def check(path: str) -> None:
        candidate = Path(os.path.abspath(path))
        if not is_under_cwd(candidate) or any(
            part.is_symlink() for part in [candidate, *candidate.parents]
        ):
            raise ValueError("MLX serving model/adapter paths must stay under cwd without symlinks")
        if not candidate.is_dir():
            raise ValueError("MLX serving requires a local model/adapter directory")
        if any(member.is_symlink() for member in candidate.iterdir()):
            raise ValueError("MLX serving model/adapter files must not be symlinks")

    check(model)
    metadata = read_native_adapter(Path(model), base)
    target = metadata["model"] if metadata else base
    if target is not None and (
        Path(target).exists() or Path(target).is_absolute() or target.startswith(".")
    ):
        check(target)


@dataclass(eq=False)
class _Job:
    options: dict[str, Any]
    cancelled: threading.Event = field(default_factory=threading.Event)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=8))
    output_timed_out: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Failure:
    error: Exception


class _RequestDisconnectedError(Exception):
    """The HTTP client left; this is not cancellation of the ASGI task."""


class NativeServerRuntime:
    """One model, one worker, at most eight waiting jobs and eight buffered increments."""

    def __init__(
        self,
        model_path: str,
        base_model: str | None = None,
        *,
        trust_remote_code: bool = False,
        queue_capacity: int = QUEUE_CAPACITY,
        loader: Callable | None = None,
        generator: Callable | None = None,
    ) -> None:
        from soup_cli.utils import mlx_generation

        if type(queue_capacity) is not int or not 1 <= queue_capacity <= QUEUE_CAPACITY:
            raise ValueError("MLX queue capacity must be between 1 and 8")
        self.model_path = model_path
        self.base_model = base_model
        self.trust_remote_code = trust_remote_code
        self.queue_capacity = queue_capacity
        self.loader = loader or mlx_generation.load_native_model
        self.generator = generator or mlx_generation.stream_native
        self._pending: deque[_Job] = deque()
        self._condition = threading.Condition()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._closed = False
        self._active: _Job | None = None
        self._thread: threading.Thread | None = None

    @property
    def ready(self) -> bool:
        return (
            self._ready.is_set()
            and self._error is None
            and not self._closed
            and self._thread is not None
            and self._thread.is_alive()
        )

    async def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("MLX server is closed")
            if self._thread is None and not self._closed:
                self._thread = threading.Thread(target=self._worker, name="soup-mlx", daemon=True)
                self._thread.start()
        await asyncio.to_thread(self._ready.wait)
        if self._error is not None:
            raise RuntimeError(f"MLX server initialization failed: {self._error}") from self._error
        if self._closed:
            raise RuntimeError("MLX server is closed")

    def submit(self, options: dict[str, Any]) -> _Job:
        with self._condition:
            if not self.ready:
                raise RuntimeError("MLX server is not ready")
            if len(self._pending) >= self.queue_capacity:
                raise OverflowError("MLX request queue is full")
            job = _Job(options)
            self._pending.append(job)
            self._condition.notify()
            return job

    def cancel(self, job: _Job) -> None:
        job.cancelled.set()
        with self._condition:
            try:
                self._pending.remove(job)
            except ValueError:
                pass  # Already decoding, or already complete.
            self._condition.notify_all()

    async def next_event(self, job: _Job, disconnected: Callable | None = None) -> Any:
        while not job.cancelled.is_set():
            if disconnected is not None and await disconnected():
                self.cancel(job)
                raise _RequestDisconnectedError
            try:
                return job.events.get_nowait()
            except queue.Empty:
                if self._error is not None:
                    return _Failure(self._error)
                await asyncio.sleep(0.01)
        raise _RequestDisconnectedError

    def _put(self, job: _Job, event: Any) -> bool:
        deadline = time.monotonic() + OUTPUT_PUT_TIMEOUT_SECONDS
        while not job.cancelled.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                job.output_timed_out.set()
                logger.warning(
                    "MLX output queue put exceeded %g-second deadline; cancelling request",
                    OUTPUT_PUT_TIMEOUT_SECONDS,
                )
                self.cancel(job)
                return False
            try:
                job.events.put(event, timeout=min(0.05, remaining))
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _clear_cache() -> None:
        # A replaced boundary in portable tests must not import MLX as a side effect.
        core = sys.modules.get("mlx.core")
        if core is not None:
            clear = getattr(core, "clear_cache", None) or core.metal.clear_cache
            clear()

    def _worker(self) -> None:
        model = tokenizer = None
        try:
            try:
                model, tokenizer = self.loader(
                    self.model_path,
                    self.base_model,
                    trust_remote_code=self.trust_remote_code,
                    require_dense=True,
                )
            except Exception as exc:
                # A loader traceback can retain an entire partially loaded model.
                # Keep the install/startup diagnostic, not native frame references.
                self._error = RuntimeError(str(exc))
                return
            finally:
                self._ready.set()
            while True:
                with self._condition:
                    while not self._pending and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    job = self._pending.popleft()
                    self._active = job
                iterator = None
                try:
                    iterator = iter(self.generator(model, tokenizer, **job.options))
                    terminal = False
                    while not job.cancelled.is_set():
                        chunk = next(iterator)
                        if not self._put(job, chunk):
                            break
                        if chunk.finish_reason is not None:
                            terminal = True
                            break
                    if not terminal and not job.cancelled.is_set():
                        raise RuntimeError("Native generator ended without a terminal increment")
                except StopIteration:
                    if not job.cancelled.is_set():
                        self._put(job, _Failure(RuntimeError("Native generator ended early")))
                except Exception as exc:
                    logger.exception("MLX generation failed")
                    error = (NativePromptError() if isinstance(exc, NativePromptError)
                             else RuntimeError())
                    self._put(job, _Failure(error))
                finally:
                    try:
                        if iterator is not None and hasattr(iterator, "close"):
                            iterator.close()
                    finally:
                        self._clear_cache()
                        with self._condition:
                            self._active = None
                            self._condition.notify_all()
        except Exception as exc:
            self._error = RuntimeError(str(exc))
            self._ready.set()
            logger.exception("MLX worker failed")
        finally:
            model = tokenizer = None
            self._clear_cache()

    async def aclose(self) -> None:
        with self._condition:
            self._closed = True
            for job in list(self._pending) + ([self._active] if self._active else []):
                job.cancelled.set()
            self._pending.clear()
            self._condition.notify_all()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 10)
            if self._thread.is_alive():
                raise RuntimeError("MLX worker did not stop at its next decode boundary")


def install_native_routes(
    app: Any,
    runtime: NativeServerRuntime,
    *,
    model_name: str,
    max_tokens_default: int,
    browser_guard: Callable,
) -> None:
    """Register only the agreed native routes; keep middleware in the shared factory."""
    from fastapi import Depends, HTTPException, Request
    from fastapi.responses import Response, StreamingResponse

    class NativeStreamingResponse(StreamingResponse):
        native_job: _Job

        async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
            async def wait_for_output_timeout() -> None:
                while not self.native_job.output_timed_out.is_set():
                    await asyncio.sleep(0.01)

            response = asyncio.create_task(super().__call__(scope, receive, send))
            timed_out = asyncio.create_task(wait_for_output_timeout())
            try:
                done, _ = await asyncio.wait(
                    (response, timed_out), return_when=asyncio.FIRST_COMPLETED,
                )
                if response in done:
                    await response
                else:
                    # The native deadline must also release a blocked ASGI send.
                    response.cancel()
                    await asyncio.gather(response, return_exceptions=True)
            finally:
                # Also runs when disconnect happens before the iterator starts.
                runtime.cancel(self.native_job)
                for task in (response, timed_out):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(response, timed_out, return_exceptions=True)

    @app.get("/health", response_model=NativeHealth)
    def health() -> dict:
        if not runtime.ready:
            raise HTTPException(503, "MLX model is not ready")
        return {"status": "ok", "model": model_name, "device": "mlx"}

    @app.get("/v1/models", response_model=NativeModels)
    def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "soup"}],
        }

    def check_failure(event: Any) -> None:
        if isinstance(event, _Failure):
            if isinstance(event.error, NativePromptError):
                raise HTTPException(422, "MLX prompt is invalid or exceeds the input token limit")
            raise HTTPException(500, "MLX generation failed")

    request_schema = NativeRequest.model_json_schema(ref_template="#/components/schemas/{model}")
    request_definitions = request_schema.pop("$defs", {})

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(browser_guard)],
        response_model=NativeCompletion,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": request_schema},
                },
            }
        },
    )
    async def completions(raw: Request) -> Any:
        body = bytearray()
        async for part in raw.stream():
            if len(body) + len(part) > MAX_BODY_BYTES:
                raise HTTPException(413, "MLX request body exceeds 1 MiB")
            body.extend(part)
        try:
            request = NativeRequest.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(
                422, [{key: error[key] for key in ("loc", "msg", "type")}
                      for error in exc.errors()]
            ) from exc
        if request.model not in {None, model_name}:
            raise HTTPException(404, "Unknown model; use GET /v1/models")
        options = dict(
            messages=[message.model_dump() for message in request.messages],
            max_tokens=request.max_tokens or max_tokens_default,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=tuple([request.stop] if isinstance(request.stop, str) else request.stop or []),
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            seed=secrets.randbits(32),
        )
        try:
            job = runtime.submit(options)
        except (OverflowError, RuntimeError) as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def usage(chunk: NativeGenerationChunk) -> dict:
            return {
                "prompt_tokens": chunk.prompt_tokens,
                "completion_tokens": chunk.completion_tokens,
                "total_tokens": chunk.prompt_tokens + chunk.completion_tokens,
            }

        def frame(delta: dict, reason: str | None = None, final: Any = None) -> str:
            value = dict(
                id=completion_id,
                object="chat.completion.chunk",
                created=created,
                model=model_name,
                choices=[dict(index=0, delta=delta, finish_reason=reason)],
            )
            if final is not None:
                value["usage"] = usage(final)
            validated = NativeStreamFrame.model_validate(value).model_dump(exclude_unset=True)
            return "data: " + json.dumps(validated, ensure_ascii=False) + "\n\n"

        try:
            first = await runtime.next_event(job, raw.is_disconnected)
            check_failure(first)
            if request.stream:

                async def stream() -> AsyncIterator[str]:
                    try:
                        yield frame({"role": "assistant"})
                        chunk = first
                        while True:
                            if isinstance(chunk, _Failure):
                                yield 'data: {"error":{"message":"MLX generation failed"}}\n\n'
                                break
                            if chunk.text:
                                yield frame({"content": chunk.text})
                            if chunk.finish_reason is not None:
                                yield frame({}, chunk.finish_reason, chunk)
                                break
                            chunk = await runtime.next_event(job, raw.is_disconnected)
                        yield "data: [DONE]\n\n"
                    except _RequestDisconnectedError:
                        return
                    except Exception:
                        logger.exception("MLX streaming response failed")
                        yield 'data: {"error":{"message":"MLX generation failed"}}\n\n'
                        yield "data: [DONE]\n\n"
                    finally:
                        runtime.cancel(job)

                response = NativeStreamingResponse(stream(), media_type="text/event-stream")
                response.native_job = job
                return response
            text = []
            chunk = first
            while True:
                check_failure(chunk)
                text.append(chunk.text)
                if chunk.finish_reason is not None:
                    break
                chunk = await runtime.next_event(job, raw.is_disconnected)
            return dict(
                id=completion_id,
                object="chat.completion",
                created=created,
                model=model_name,
                choices=[
                    dict(
                        index=0,
                        message=dict(role="assistant", content="".join(text)),
                        finish_reason=chunk.finish_reason,
                    )
                ],
                usage=usage(chunk),
            )
        except _RequestDisconnectedError:
            return Response(status_code=204)
        finally:
            if not request.stream or sys.exc_info()[0] is not None:
                runtime.cancel(job)

    original_openapi = app.openapi

    def native_openapi() -> dict:
        schema = original_openapi()
        schema.setdefault("components", {}).setdefault("schemas", {}).update(request_definitions)
        return schema

    app.openapi = native_openapi
