"""Portable contracts for the native server; replace only the native boundary."""

import asyncio
import importlib.util
import inspect
import json
import logging
import threading
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def test_public_native_server_uses_existing_app_factory(tmp_path, monkeypatch):
    import uvicorn

    from soup_cli.cli import app
    from soup_cli.utils import mlx_generation

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: True if name in {"mlx", "mlx_lm"} else find_spec(name),
    )

    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "llama"}))
    calls = []

    def load(path, base_model=None, **kwargs):
        calls.append((path, base_model, kwargs))
        return SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr(mlx_generation, "load_native_model", load)
    served = []

    def run(application, **kwargs):
        with TestClient(application) as client:
            assert client.get("/health").json()["status"] == "ok"
            assert client.get("/v1/models").json()["data"][0]["id"] == "base"
            assert client.post("/v1/tools/bash", json={"command": "echo x"}).status_code == 404
            served.append(application)

    monkeypatch.setattr(uvicorn, "run", run)
    result = CliRunner().invoke(app, ["serve", "--backend", "mlx", "--model", str(base)])
    assert result.exit_code == 0, result.output
    assert len(served) == len(calls) == 1
    assert calls[0][2]["require_dense"] is True


def _application(generator=None, loader=None, queue_capacity=8, **kwargs):
    from soup_cli.commands.serve import _create_app
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk
    from soup_cli.utils.mlx_serve import NativeServerRuntime

    def generate(model, tokenizer, *, messages, **options):
        yield Chunk(messages[-1]["content"], 3, 1)
        yield Chunk("!", 3, 2, "length")

    runtime = NativeServerRuntime(
        "base",
        loader=loader or (lambda *args, **kw: (object(), object())),
        generator=generator or generate,
        queue_capacity=queue_capacity,
    )
    return _create_app(
        None,
        None,
        "mlx",
        "fixed",
        32,
        native_runtime=runtime,
        **kwargs,
    ), runtime


def _body(**changes):
    return {"messages": [{"role": "user", "content": "héllo"}], **changes}


def _frames(text):
    return [
        json.loads(line[6:]) if line != "data: [DONE]" else "DONE"
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def test_only_fixed_model_routes_and_real_usage():
    app, runtime = _application()
    with TestClient(app) as client:
        result = client.post("/v1/chat/completions", json=_body()).json()
        assert result["model"] == "fixed"
        assert result["choices"] == [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "héllo!"},
                "finish_reason": "length",
            }
        ]
        assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        streamed = client.post("/v1/chat/completions", json=_body(stream=True))
        assert streamed.headers["content-type"].startswith("text/event-stream")
        frames = _frames(streamed.text)
        assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert "".join(f["choices"][0]["delta"].get("content", "") for f in frames[:-1]) == "héllo!"
        assert frames[-2]["choices"][0]["finish_reason"] == "length"
        assert frames[-2]["usage"] == result["usage"]
        assert frames[-1] == "DONE"
        assert len({f["id"] for f in frames[:-1]}) == 1
        for path in [
            "/metrics",
            "/v1/adapters",
            "/v1/completions",
            "/v1/messages",
            "/v1/tools/python",
            "/v1/tools/bash",
            "/v1/output_constraint",
        ]:
            assert client.get(path).status_code == 404
            assert client.post(path, json={}).status_code == 404
    assert not runtime.ready
    assert not runtime._thread.is_alive()


@pytest.mark.parametrize(
    "changes",
    [
        {"messages": []},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        {"messages": [{"role": "user", "content": "x", "tool_calls": []}]},
        {"messages": [{"role": "user", "content": "x" * 65537}]},
        {"messages": [{"role": "user", "content": "x"}] * 129},
        {"temperature": -1},
        {"temperature": 2.1},
        {"temperature": True},
        {"top_p": -0.1},
        {"top_p": 1.1},
        {"top_p": True},
        {"max_tokens": 0},
        {"max_tokens": 16385},
        {"max_tokens": True},
        {"stream": "true"},
        {"stop": ""},
        {"stop": ["a"] * 5},
        {"stop": "x" * 257},
        {"stop": [3]},
        {"tools": []},
        {"logprobs": True},
        {"n": 2},
        {"adapter": "other"},
        {"seed": 0},
        {"response_format": {"type": "json_object"}},
        {"stream_options": {"include_usage": True}},
        {"model_path": "../../private"},
    ],
)
def test_unsupported_requests_refused_before_decoding(changes):
    called = []

    def generate(*args, **kwargs):
        called.append(True)
        raise AssertionError("invalid request must not reach decoding")

    app, _ = _application(generator=generate)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_body(**changes))
        assert response.status_code == 422, response.text
    assert not called


def test_unknown_model_malformed_json_nonfinite_and_body_cap():
    from soup_cli.utils.mlx_serve import MAX_BODY_BYTES

    app, _ = _application()
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=_body(model="other")).status_code == 404
        for body in [b"{", b"{}", b'{"messages":[],"temperature":NaN}']:
            assert client.post("/v1/chat/completions", content=body).status_code == 422
        response = client.post("/v1/chat/completions", content=b"x" * (MAX_BODY_BYTES + 1))
        assert response.status_code == 413
        assert client.post("/v1/chat/completions", json=_body()).status_code == 200


def test_origin_and_cors_policy_preserved_without_rejecting_proxy_host():
    app, _ = _application()
    with TestClient(app, base_url="http://proxy.example") as client:
        assert client.post("/v1/chat/completions", json=_body()).status_code == 200
        response = client.post(
            "/v1/chat/completions", json=_body(), headers={"Origin": "https://evil.example"}
        )
        assert response.status_code == 403
        allowed = client.post(
            "/v1/chat/completions", json=_body(), headers={"Origin": "http://localhost:3000"}
        )
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_worker_owns_load_decode_and_request_seeds(monkeypatch):
    from soup_cli.utils import mlx_serve
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    records = []
    seeds = iter([123, 456])
    monkeypatch.setattr(mlx_serve.secrets, "randbits", lambda bits: next(seeds))

    def load(*args, **kwargs):
        records.append(("load", threading.get_ident(), kwargs))
        return object(), object()

    def generate(model, tokenizer, **options):
        records.append(("decode", threading.get_ident(), options))
        yield Chunk(options["messages"][0]["content"], 4, 1, "stop")

    app, _ = _application(loader=load, generator=generate)
    with TestClient(app) as client:
        for content in ["first", "unrelated"]:
            response = client.post(
                "/v1/chat/completions",
                json={
                    **_body(temperature=0, top_p=0.3, max_tokens=7, stop=["END"]),
                    "messages": [{"role": "user", "content": content}],
                },
            )
            assert response.json()["choices"][0]["message"]["content"] == content
    assert [r[0] for r in records] == ["load", "decode", "decode"]
    assert len({r[1] for r in records}) == 1
    assert records[0][1] != threading.get_ident()
    assert [r[2]["seed"] for r in records[1:]] == [123, 456]
    for _, _, options in records[1:]:
        assert options["max_tokens"] == 7 and options["top_p"] == 0.3
        assert options["temperature"] == 0 and options["stop"] == ("END",)
        assert options["max_prompt_tokens"] == 16384


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("after_first", [False, True])
def test_generation_errors_are_sanitized_and_next_request_works(stream, after_first):
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    def generate(model, tokenizer, **options):
        if options["messages"][0]["content"] == "bad":
            if after_first:
                yield Chunk("partial", 3, 1)
            raise RuntimeError("/private/token-secret.txt")
        yield Chunk("recovered", 3, 2, "stop")

    app, _ = _application(generator=generate)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "bad"}],
                "stream": stream,
            },
        )
        assert "token-secret" not in response.text
        if stream and after_first:
            assert response.status_code == 200
            assert any("error" in f for f in _frames(response.text) if isinstance(f, dict))
            assert _frames(response.text)[-1] == "DONE"
        else:
            assert response.status_code == 500
        assert client.post("/v1/chat/completions", json=_body()).status_code == 200


def test_prompt_failure_is_422_and_startup_failure_cleans_worker():
    from soup_cli.utils.mlx_generation import NativePromptError

    def invalid(*args, **options):
        raise NativePromptError("sensitive model path")

    app, _ = _application(generator=invalid)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_body())
        assert response.status_code == 422
        assert "sensitive" not in response.text

    def broken(*args, **kwargs):
        raise ImportError("Install soup-cli[mlx] on Apple Silicon")

    app, runtime = _application(loader=broken)
    client = TestClient(app)
    assert client.get("/health").status_code == 503
    client.close()
    with pytest.raises(RuntimeError, match=r"soup-cli\[mlx\]"):
        with TestClient(app):
            pytest.fail("startup must fail")
    assert not runtime.ready and not runtime._thread.is_alive()


async def _asgi_post(app, body, *, on_body=None, on_start=None, incoming=None):
    """Capture ASGI frames as sent, without TestClient's streaming buffer."""
    incoming = incoming or asyncio.Queue()
    incoming.put_nowait(
        {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
    )
    sent = []

    async def receive():
        return await incoming.get()

    async def send(event):
        if event["type"] == "http.response.start" and on_start is not None:
            on_start(event)
        sent.append(event)
        if event["type"] == "http.response.body" and on_body is not None:
            result = on_body(event.get("body", b""), incoming)
            if inspect.isawaitable(result):
                await result

    scope = dict(
        type="http",
        asgi={"version": "3.0", "spec_version": "2.0"},
        http_version="1.1",
        method="POST",
        scheme="http",
        path="/v1/chat/completions",
        raw_path=b"/v1/chat/completions",
        query_string=b"",
        root_path="",
        headers=[(b"host", b"localhost"), (b"content-type", b"application/json")],
        client=("127.0.0.1", 12),
        server=("127.0.0.1", 8000),
        state={},
    )
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    return sent


def test_first_content_frame_is_sent_before_decode_finishes():
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    content_seen = threading.Event()
    closed = threading.Event()

    def generate(*args, **options):
        try:
            yield Chunk("first é", 3, 1)
            assert content_seen.wait(3), "content was buffered until decoding finished"
            yield Chunk(" second", 3, 2, "stop")
        finally:
            closed.set()

    async def run():
        app, _ = _application(generator=generate)
        async with app.router.lifespan_context(app):

            def sent(body, incoming):
                if "first é" in body.decode():
                    content_seen.set()

            events = await _asgi_post(app, _body(stream=True), on_body=sent)
            text = b"".join(e.get("body", b"") for e in events).decode()
            assert content_seen.is_set()
            assert (
                "".join(f["choices"][0]["delta"].get("content", "") for f in _frames(text)[:-1])
                == "first é second"
            )
            assert closed.wait(1)

    asyncio.run(run())


def test_stream_disconnect_closes_decode_and_next_request_runs():
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    closed = threading.Event()
    steps = []

    def generate(*args, **options):
        if options["messages"][0]["content"] == "next":
            yield Chunk("next", 2, 1, "stop")
            return
        try:
            for index in range(1000):
                steps.append(index)
                yield Chunk("increment", 2, index + 1)
        finally:
            closed.set()

    async def run():
        app, _ = _application(generator=generate)
        async with app.router.lifespan_context(app):
            disconnected = False

            def disconnect(body, incoming):
                nonlocal disconnected
                if b"increment" in body and not disconnected:
                    disconnected = True
                    incoming.put_nowait({"type": "http.disconnect"})

            await _asgi_post(app, _body(stream=True), on_body=disconnect)
            assert await asyncio.to_thread(closed.wait, 2), "decoder survived disconnection"
            assert len(steps) < 1000
            later = await _asgi_post(app, {"messages": [{"role": "user", "content": "next"}]})
            payload = json.loads(b"".join(e.get("body", b"") for e in later))
            assert payload["choices"][0]["message"]["content"] == "next"

    asyncio.run(run())


def test_bounded_queue_503_and_queued_cancellation_releases_capacity():
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    entered, release = threading.Event(), threading.Event()
    decoded = []

    def generate(*args, **options):
        decoded.append(options["label"])
        if options["label"] == "active":
            entered.set()
            assert release.wait(3)
        yield Chunk(options["label"], 2, 1, "stop")

    app, runtime = _application(generator=generate, queue_capacity=1)
    with TestClient(app) as client:
        active = runtime.submit({"label": "active"})
        assert entered.wait(2)
        pending = runtime.submit({"label": "cancelled"})
        try:
            busy = client.post("/v1/chat/completions", json=_body())
            assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
            assert client.get("/health").status_code == 200
            runtime.cancel(pending)
            replacement = runtime.submit({"label": "replacement"})
            runtime.cancel(active)
            release.set()
            chunk = asyncio.run(asyncio.wait_for(runtime.next_event(replacement), 2))
            assert chunk.text == "replacement" and chunk.finish_reason == "stop"
            assert decoded == ["active", "replacement"]
        finally:
            release.set()
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.submit({})


def test_nonstream_disconnect_removes_waiting_request():
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    entered, release = threading.Event(), threading.Event()

    def generate(*args, **options):
        entered.set()
        assert release.wait(3)
        yield Chunk("finished", 2, 1, "stop")

    async def run():
        app, runtime = _application(generator=generate, queue_capacity=1)
        async with app.router.lifespan_context(app):
            active = runtime.submit({})
            assert await asyncio.to_thread(entered.wait, 2)
            incoming = asyncio.Queue()
            waiting = asyncio.create_task(_asgi_post(app, _body(), incoming=incoming))
            for _ in range(100):
                if runtime._pending:
                    break
                await asyncio.sleep(0.01)
            assert len(runtime._pending) == 1
            incoming.put_nowait({"type": "http.disconnect"})
            try:
                events = await waiting
                assert events[0]["status"] == 204
                assert b"".join(event.get("body", b"") for event in events) == b""
                assert len(runtime._pending) == 0
            finally:
                runtime.cancel(active)
                release.set()

    asyncio.run(run())


def test_closing_an_unstarted_runtime_cannot_hang_start():
    async def run():
        _, runtime = _application()
        await runtime.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await asyncio.wait_for(runtime.start(), 1)

    asyncio.run(run())


def test_failed_startup_releases_allocations_even_while_app_stays_alive():
    import gc
    import weakref

    references = []

    class Allocation:
        pass

    def broken(*args, **kwargs):
        allocated = Allocation()
        references.append(weakref.ref(allocated))
        raise ValueError("failed after allocating a model")

    app, runtime = _application(loader=broken)
    with pytest.raises(RuntimeError, match="after allocating"):
        with TestClient(app):
            pytest.fail("startup should fail")
    gc.collect()
    assert not runtime.ready
    assert references[0]() is None, "saved exception traceback retained the failed model"


def test_zero_top_p_has_an_explicit_refusal():
    app, _ = _application()
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_body(top_p=0))
        assert response.status_code == 422
        assert "top_p" in response.text


def test_native_openapi_exposes_request_and_response_schemas():
    app, _ = _application()
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/v1/chat/completions"]["post"]
    assert "requestBody" in operation, "native API must expose its validated request schema"
    body = operation["requestBody"]["content"]["application/json"]["schema"]
    assert body["additionalProperties"] is False
    assert body["properties"]["messages"]["maxItems"] == 128
    assert (
        "NativeCompletion"
        in operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    )
    assert "NativeMessage" in schema["components"]["schemas"]


@pytest.mark.parametrize("stream", [False, True])
def test_invalid_engine_usage_cannot_escape_the_response_schema(stream):
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    def broken(*args, **kwargs):
        yield Chunk("text", -1, 1, "stop")

    app, _ = _application(generator=broken)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions", json=_body(stream=stream))
        if stream:
            assert any(
                "error" in frame for frame in _frames(response.text) if isinstance(frame, dict)
            )
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            assert response.status_code == 500


@pytest.mark.parametrize(
    "flags, diagnostic",
    [
        (["--device", "cpu"], "--device"),
        (["--hub", "modelscope"], "--hub"),
        (["--tensor-parallel", "2"], "--tensor-parallel"),
        (["--gpu-memory", "0.8"], "--gpu-memory"),
        (["--max-model-len", "128"], "--max-model-len"),
        (["--speculative-decoding", "draft"], "--speculative-decoding"),
        (["--num-speculative-tokens", "3"], "--num-speculative-tokens"),
        (["--adapters", "other=./adapter"], "--adapters"),
        (["--prefix-cache"], "--prefix-cache"),
        (["--auto-spec"], "--auto-spec"),
        (["--structured-output", "json"], "--structured-output"),
        (["--json-schema", "file.json"], "--json-schema"),
        (["--regex-pattern", "x"], "--regex-pattern"),
        (["--dashboard"], "--dashboard"),
        (["--trace"], "--trace"),
        (["--trace-endpoint", "https://example.org"], "--trace-endpoint"),
        (["--auto-quant"], "--auto-quant"),
        (["--trace-log", "out.jsonl"], "--trace-log"),
        (["--trace-log-cap-mb", "1"], "--trace-log-cap-mb"),
        (["--record-thumbs", "out.db"], "--record-thumbs"),
        (["--reasoning-parser", "qwen3"], "--reasoning-parser"),
        (["--steer", "x"], "--steer"),
        (["--steer-strength", "2"], "--steer-strength"),
        (["--bank", "x"], "--bank"),
        (["--bank-strength", "2"], "--bank-strength"),
        (["--mole", "x"], "--mole"),
        (["--kv-cache-type", "bf16"], "--kv-cache-type"),
        (["--max-tokens", "0"], "max_tokens"),
    ],
)
def test_native_cli_refuses_unimplemented_flags_before_loading(
    tmp_path, monkeypatch, flags, diagnostic
):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    result = CliRunner().invoke(app, ["serve", "--backend", "mlx", "--model", str(base), *flags])
    assert result.exit_code == 2, result.output
    assert diagnostic in result.output
    assert "Loading model" not in result.output


@pytest.mark.parametrize("force_color", [False, True])
def test_native_cli_keeps_nonloopback_token_rule(tmp_path, monkeypatch, force_color):
    from rich.console import Console

    from soup_cli.cli import app
    from soup_cli.commands import serve
    from tests.conftest import strip_ansi

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(
        serve, "console", Console(
            force_terminal=force_color, no_color=False, color_system="standard", width=400
        )
    )
    result = CliRunner().invoke(
        app, ["serve", "--backend", "mlx", "--model", "missing", "--host", "0.0.0.0"],
        color=force_color,
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    if force_color:
        assert "\x1b[" in result.output
    assert "requires --tool-auth-token" in " ".join(strip_ansi(result.output).split())


@pytest.mark.requires_symlink
def test_native_paths_refuse_escape_symlinks_and_conflicting_base(tmp_path, monkeypatch):
    from soup_cli.utils.mlx_serve import validate_serve_paths

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    base = cwd / "base"
    base.mkdir()
    monkeypatch.chdir(cwd)
    validate_serve_paths(str(base), None)
    for invalid in [outside, cwd / "missing"]:
        with pytest.raises(ValueError):
            validate_serve_paths(str(invalid), None)
    link = cwd / "link"
    link.symlink_to(base, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_serve_paths(str(link), None)
    with pytest.raises(ValueError, match="only valid"):
        validate_serve_paths(str(base), "org/other")


def test_a_stream_whose_body_never_starts_still_cancels_its_decode():
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    closed = threading.Event()

    def generate(*args, **options):
        if options["messages"][0]["content"] == "next":
            yield Chunk("next", 2, 1, "stop")
            return
        try:
            for index in range(1000):
                yield Chunk("increment", 2, index + 1)
        finally:
            closed.set()

    def disconnect(event):
        raise OSError("client left before the first byte")

    async def run():
        app, _ = _application(generator=generate)
        async with app.router.lifespan_context(app):
            with pytest.raises(Exception):
                await _asgi_post(app, _body(stream=True), on_start=disconnect)
            assert await asyncio.to_thread(closed.wait, 2), "decode outlived its stream"
            later = await _asgi_post(app, {"messages": [{"role": "user", "content": "next"}]})
            payload = json.loads(b"".join(event.get("body", b"") for event in later))
            assert payload["choices"][0]["message"]["content"] == "next"

    asyncio.run(run())


def test_missing_native_dependency_is_refused_before_startup(tmp_path, monkeypatch):
    import sys

    import uvicorn

    from soup_cli.cli import app
    from soup_cli.utils import mlx_generation
    from tests.conftest import strip_ansi

    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setitem(sys.modules, "mlx_lm", None)

    def unexpected(*args, **kwargs):
        pytest.fail("missing dependency must be refused before loading or starting uvicorn")

    monkeypatch.setattr(uvicorn, "run", unexpected)
    monkeypatch.setattr(mlx_generation, "load_native_model", unexpected)
    result = CliRunner().invoke(app, ["serve", "--backend", "mlx", "--model", str(base)])
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 1, (text, repr(result.exception))
    assert 'pip install "soup-cli[mlx,serve]"' in text
    assert "Starting server" not in text and "Loading model" not in text


@pytest.mark.parametrize("invalid", [{"stop": [""]}, {"temperature": "private-secret"}])
def test_validation_errors_only_expose_portable_public_details(invalid):
    app, _ = _application()
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_body(**invalid))
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert errors and all(set(error) == {"loc", "msg", "type"} for error in errors)
        assert "private-secret" not in response.text


def test_engine_value_error_is_not_reported_as_a_bad_prompt():
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    def generate(*args, **options):
        if options["messages"][0]["content"] == "bad":
            raise ValueError("private kernel shape mismatch")
        yield Chunk("recovered", 3, 1, "stop")

    app, _ = _application(generator=generate)
    with TestClient(app) as client:
        bad = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "bad"}]}
        )
        assert bad.status_code == 500
        assert bad.json()["detail"] == "MLX generation failed"
        assert "private" not in bad.text
        assert client.post("/v1/chat/completions", json=_body()).status_code == 200


def test_cancelled_decode_does_not_advance_the_iterator_again(monkeypatch):
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    steps = []
    closed = threading.Event()

    def generate(*args, **options):
        try:
            for index in range(3):
                steps.append(index)
                yield Chunk("increment", 2, index + 1)
        finally:
            closed.set()

    async def run():
        app, runtime = _application(generator=generate)
        put = runtime._put

        def cancel_after_put(job, event):
            result = put(job, event)
            runtime.cancel(job)
            return result

        monkeypatch.setattr(runtime, "_put", cancel_after_put)
        async with app.router.lifespan_context(app):
            runtime.submit({})
            assert await asyncio.to_thread(closed.wait, 2), "cancelled iterator was not closed"
            assert steps == [0], "decode advanced after cancellation"

    asyncio.run(run())


def test_simultaneous_http_streams_keep_outputs_and_usage_separate(monkeypatch):
    from soup_cli.utils import mlx_serve
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    entered, release = threading.Event(), threading.Event()
    seeds = iter([123, 456])
    monkeypatch.setattr(mlx_serve.secrets, "randbits", lambda bits: next(seeds))
    observed = []

    def generate(*args, **options):
        content = options["messages"][0]["content"]
        observed.append((content, options["seed"]))
        if content == "first é":
            entered.set()
            assert release.wait(3), "concurrent request never reached the HTTP queue"
        yield Chunk(content, 3, 1)
        yield Chunk("!", 3, 2, "length")

    async def run():
        app, runtime = _application(generator=generate)
        async with app.router.lifespan_context(app):
            first = asyncio.create_task(_asgi_post(
                app, {"messages": [{"role": "user", "content": "first é"}], "stream": True}
            ))
            assert await asyncio.to_thread(entered.wait, 2)
            second = asyncio.create_task(_asgi_post(
                app, {"messages": [{"role": "user", "content": "second 中"}], "stream": True}
            ))
            try:
                for _ in range(100):
                    if runtime._pending:
                        break
                    await asyncio.sleep(0.01)
                assert len(runtime._pending) == 1
            finally:
                release.set()
            responses = await asyncio.gather(first, second)
            identities = []
            for events, expected in zip(responses, ["first é!", "second 中!"], strict=True):
                assert events[0]["status"] == 200
                frames = _frames(b"".join(e.get("body", b"") for e in events).decode())
                assert frames[-1] == "DONE"
                assert "".join(f["choices"][0]["delta"].get("content", "")
                               for f in frames[:-1]) == expected
                assert frames[-2]["usage"] == {
                    "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
                }
                identities.append(frames[0]["id"])
            assert len(set(identities)) == 2
            assert observed == [("first é", 123), ("second 中", 456)]

    asyncio.run(run())


def test_a_streamed_client_that_leaves_ends_without_an_error_frame(monkeypatch, caplog):
    from starlette.requests import Request

    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    left, closed = threading.Event(), threading.Event()

    async def is_disconnected(self):
        return left.is_set()

    # Starlette's listener sees no disconnect; only the route's poll does.
    monkeypatch.setattr(Request, "is_disconnected", is_disconnected)

    def generate(*args, **options):
        try:
            for index in range(1000):
                yield Chunk("increment", 2, index + 1)
        finally:
            closed.set()

    def leave(body, incoming):
        if b"increment" in body:
            left.set()

    async def run():
        app, _ = _application(generator=generate)
        async with app.router.lifespan_context(app):
            events = await _asgi_post(app, _body(stream=True), on_body=leave)
            assert await asyncio.to_thread(closed.wait, 2), "decode outlived its stream"
            return b"".join(event.get("body", b"") for event in events).decode()

    with caplog.at_level(logging.ERROR, logger="soup_cli.utils.mlx_serve"):
        text = asyncio.run(run())
    assert not any(isinstance(frame, dict) and "error" in frame for frame in _frames(text))
    assert "[DONE]" not in text
    assert "MLX streaming response failed" not in caplog.text


@pytest.mark.requires_symlink
@pytest.mark.parametrize("artifact", ["model", "adapter"])
def test_native_paths_refuse_a_symlinked_file_inside_the_artifact(
    tmp_path, monkeypatch, artifact,
):
    from soup_cli.utils.mlx_serve import validate_serve_paths

    cwd, outside = tmp_path / "cwd", tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    base = cwd / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    monkeypatch.chdir(cwd)
    target = base
    if artifact == "adapter":
        target = cwd / "adapter"
        target.mkdir()
        (target / "adapters.safetensors").write_bytes(b"")
        _write_native_metadata(target, base)
    validate_serve_paths(str(target), None)
    name = "config.json" if artifact == "model" else "adapters.safetensors"
    (outside / name).write_bytes(b"")
    (target / name).unlink()
    (target / name).symlink_to(outside / name)
    with pytest.raises(ValueError, match="must not be symlinks"):
        validate_serve_paths(str(target), None)


def _write_native_metadata(adapter, base):
    (adapter / "adapter_config.json").write_text(json.dumps({
        "model": str(base), "fine_tune_type": "lora", "num_layers": 1,
        "lora_parameters": {
            "rank": 4, "scale": 2.0, "dropout": 0.0, "keys": ["self_attn.q_proj"],
        },
    }))


def test_native_paths_refuse_an_adapter_whose_metadata_base_is_outside_cwd(tmp_path, monkeypatch):
    from soup_cli.utils.mlx_serve import validate_serve_paths

    cwd, outside = tmp_path / "cwd", tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    base, adapter = cwd / "base", cwd / "adapter"
    base.mkdir()
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"")
    monkeypatch.chdir(cwd)
    _write_native_metadata(adapter, base)
    validate_serve_paths(str(adapter), None)
    _write_native_metadata(adapter, outside)
    with pytest.raises(ValueError, match="under cwd"):
        validate_serve_paths(str(adapter), None)


def test_a_connected_reader_that_stops_reading_cannot_hold_the_worker(monkeypatch):
    from soup_cli.utils import mlx_serve
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    # The production timeout is documented separately; use a short test budget.
    monkeypatch.setattr(mlx_serve, "OUTPUT_PUT_TIMEOUT_SECONDS", 0.5)
    closed = threading.Event()

    def generate(*args, **options):
        if options["messages"][0]["content"] == "next":
            yield Chunk("next", 2, 1, "stop")
            return
        try:
            for index in range(1000):
                yield Chunk("increment", 2, index + 1)
        finally:
            closed.set()

    async def run():
        blocked, send_closed = asyncio.Event(), asyncio.Event()

        async def stop_reading(body, incoming):
            blocked.set()
            try:
                await asyncio.Future()  # Connected, without draining any more body frames.
            finally:
                send_closed.set()

        app, runtime = _application(generator=generate)
        async with app.router.lifespan_context(app):
            slow = asyncio.create_task(_asgi_post(app, _body(stream=True), on_body=stop_reading))
            try:
                await asyncio.wait_for(blocked.wait(), 2)
                events = await asyncio.wait_for(_asgi_post(
                    app, {"messages": [{"role": "user", "content": "next"}]}
                ), 2)
                payload = json.loads(b"".join(event.get("body", b"") for event in events))
                assert payload["choices"][0]["message"]["content"] == "next"
                assert closed.is_set(), "stalled decode was not closed before the next request"
                await asyncio.wait_for(slow, 2)
                assert send_closed.is_set(), "blocked ASGI send survived output timeout"
                assert not runtime._pending
            finally:
                slow.cancel()
                await asyncio.gather(slow, return_exceptions=True)

    asyncio.run(run())


def test_put_timeout_cancels_the_job_without_enqueuing_more_output(monkeypatch, caplog):
    from soup_cli.utils import mlx_serve

    monkeypatch.setattr(mlx_serve, "OUTPUT_PUT_TIMEOUT_SECONDS", 0.01)
    _, runtime = _application()
    job = mlx_serve._Job({})
    for _ in range(job.events.maxsize):
        job.events.put_nowait(object())
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.mlx_serve"):
        assert runtime._put(job, object()) is False
        assert runtime._put(job, object()) is False  # No repeated warning after cancellation.
    assert job.cancelled.is_set(), "put deadline must cancel the job itself"
    assert job.output_timed_out.is_set()
    assert job.events.qsize() == job.events.maxsize
    assert [record.getMessage() for record in caplog.records
            if record.name == "soup_cli.utils.mlx_serve"] == [
        "MLX output queue put exceeded 0.01-second deadline; cancelling request",
    ]


def test_a_reader_that_keeps_reading_slowly_gets_the_whole_stream(monkeypatch):
    from soup_cli.utils import mlx_serve
    from soup_cli.utils.mlx_generation import NativeGenerationChunk as Chunk

    assert mlx_serve.OUTPUT_PUT_TIMEOUT_SECONDS == 30.0
    budget = 0.5
    monkeypatch.setattr(mlx_serve, "OUTPUT_PUT_TIMEOUT_SECONDS", budget)
    count = 40

    def generate(*args, **options):
        for index in range(1, count):
            yield Chunk("x", 2, index)
        yield Chunk("x", 2, count, "stop")

    async def read_slowly(body, incoming):
        await asyncio.sleep(0.025)

    async def run():
        app, _ = _application(generator=generate)
        async with app.router.lifespan_context(app):
            start = asyncio.get_running_loop().time()
            events = await _asgi_post(
                app, _body(stream=True, max_tokens=count), on_body=read_slowly,
            )
            elapsed = asyncio.get_running_loop().time() - start
            return b"".join(event.get("body", b"") for event in events).decode(), elapsed

    text, elapsed = asyncio.run(run())
    frames = _frames(text)
    assert elapsed > budget, "the healthy stream must outlive one put deadline"
    assert "".join(frame["choices"][0]["delta"].get("content", "")
                   for frame in frames[:-1]) == "x" * count
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"
    assert frames[-2]["usage"] == {
        "prompt_tokens": 2, "completion_tokens": count, "total_tokens": count + 2,
    }
    assert frames[-1] == "DONE"
