"""``soup serve``: the vLLM, SGLang and MII apps check ``Origin`` on their generation routes.

The transformers app refuses a generation request whose ``Origin`` names another site
with 403 (``tests/test_serve_local_request_guard.py``). The other three apps apply the
same rule, ``utils.local_request_guard.check_browser_origin`` against the address
``soup serve`` bound, to their generation routes:

* vLLM: ``/v1/chat/completions`` and ``/v1/messages``;
* SGLang and MII: ``/v1/chat/completions``.

A request with no ``Origin`` (curl, an SDK, a reverse proxy) is
served as before, and so is one whose ``Origin`` names the bound server. ``Host`` is not
checked, so a proxy can front these routes under its own name. A refused request never
reaches the engine. ``/health`` and ``/v1/models`` are unchanged.

The engines are stand-ins, as in the other tests of these apps, so this runs anywhere.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

FOREIGN = ["https://example.org", "http://127.0.0.1.example.org", "null", "file://x"]
LOOPBACK_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:3000", "http://[::1]:8080"]

CHAT_BODY = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
MESSAGES_BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 16,
}


# ---------------------------------------------------------------------------
# Stand-in engines
# ---------------------------------------------------------------------------


class _VllmCompletion:
    def __init__(self) -> None:
        self.text = "hello"
        self.token_ids = [1, 2]
        self.finish_reason = "stop"


class _VllmRequestOutput:
    def __init__(self) -> None:
        self.outputs = [_VllmCompletion()]
        self.prompt_token_ids = [1, 2, 3]


def _vllm_app(calls: list, **kwargs):
    from soup_cli.utils.vllm import create_vllm_app

    engine = MagicMock()

    def _generate(prompt, sampling_params, request_id, **_kwargs):
        calls.append(prompt)

        async def _outputs():
            yield _VllmRequestOutput()

        return _outputs()

    engine.generate = _generate
    vllm_stub = MagicMock()
    with patch.dict(
        sys.modules,
        {"vllm": vllm_stub, "vllm.lora": MagicMock(), "vllm.lora.request": MagicMock()},
    ):
        return create_vllm_app(
            engine=engine,
            engine_model_name="test-model",
            model_name="test-model",
            max_tokens_default=16,
            **kwargs,
        )


def _sglang_app(calls: list, **kwargs):
    from soup_cli.utils.sglang import create_sglang_app

    runtime = MagicMock()
    runtime.url = "http://127.0.0.1:30000"

    def _generate(prompt, **_kwargs):
        calls.append(prompt)
        return json.dumps(
            {
                "text": "hello",
                "meta_info": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "finish_reason": {"type": "stop"},
                },
            }
        )

    runtime.generate = _generate
    return create_sglang_app(
        runtime=runtime,
        runtime_model_name="test-model",
        model_name="test-model",
        max_tokens_default=16,
        **kwargs,
    )


class _MiiResponse:
    generated_text = "hello"


def _mii_app(calls: list, **kwargs):
    from soup_cli.utils.mii import build_mii_app

    def _pipeline(prompts, max_new_tokens=None, **_kwargs):
        calls.append(prompts[0])
        return [_MiiResponse()]

    return build_mii_app(_pipeline, model_name="test-model", **kwargs)


# (backend id, app factory, route, body) for every generation route of the three apps.
ROUTES = [
    ("vllm", _vllm_app, "/v1/chat/completions", CHAT_BODY),
    ("vllm", _vllm_app, "/v1/messages", MESSAGES_BODY),
    ("sglang", _sglang_app, "/v1/chat/completions", CHAT_BODY),
    ("mii", _mii_app, "/v1/chat/completions", CHAT_BODY),
]
ROUTE_IDS = [f"{backend}{route}" for backend, _, route, _ in ROUTES]


def _client(factory, calls, *, base_url="http://127.0.0.1:8000", **kwargs) -> TestClient:
    return TestClient(factory(calls, **kwargs), base_url=base_url)


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("backend", "factory", "route", "body"), ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize("origin", FOREIGN)
def test_a_foreign_origin_is_refused_before_the_engine_runs(
    backend, factory, route, body, origin
):
    calls: list = []
    resp = _client(factory, calls).post(route, json=body, headers={"Origin": origin})

    assert resp.status_code == 403, (backend, route, resp.text)
    assert resp.json() == {"detail": "Origin not allowed"}
    assert calls == [], f"{backend} {route} ran the engine before the Origin was checked"


@pytest.mark.parametrize(("backend", "factory", "route", "body"), ROUTES, ids=ROUTE_IDS)
def test_a_request_without_origin_is_served(backend, factory, route, body):
    """curl, an SDK and a reverse proxy send no Origin, under any Host."""
    calls: list = []
    resp = _client(factory, calls, base_url="http://proxy.example").post(route, json=body)

    assert resp.status_code == 200, (backend, route, resp.text)
    assert len(calls) == 1


@pytest.mark.parametrize(("backend", "factory", "route", "body"), ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize("origin", LOOPBACK_ORIGINS)
def test_a_loopback_origin_is_served_on_a_loopback_bind(backend, factory, route, body, origin):
    calls: list = []
    resp = _client(factory, calls).post(route, json=body, headers={"Origin": origin})

    assert resp.status_code == 200, (backend, route, resp.text)
    assert len(calls) == 1


@pytest.mark.parametrize(("backend", "factory", "route", "body"), ROUTES, ids=ROUTE_IDS)
def test_a_named_bind_accepts_its_own_name_only(backend, factory, route, body):
    calls: list = []
    client = _client(factory, calls, base_url="http://10.0.0.7:8000", host="10.0.0.7")

    same = client.post(route, json=body, headers={"Origin": "http://10.0.0.7:8000"})
    loopback = client.post(route, json=body, headers={"Origin": "http://localhost:5173"})

    assert same.status_code == 200, (backend, route, same.text)
    assert loopback.status_code == 403, (backend, route, loopback.text)
    assert len(calls) == 1


@pytest.mark.parametrize(("backend", "factory", "route", "body"), ROUTES, ids=ROUTE_IDS)
def test_a_wildcard_bind_compares_origin_with_host(backend, factory, route, body):
    calls: list = []
    client = _client(factory, calls, base_url="http://10.0.0.7:8000", host="0.0.0.0")

    same = client.post(route, json=body, headers={"Origin": "http://10.0.0.7:3000"})
    foreign = client.post(route, json=body, headers={"Origin": "https://example.org"})

    assert same.status_code == 200, (backend, route, same.text)
    assert foreign.status_code == 403, (backend, route, foreign.text)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "factory", [_vllm_app, _sglang_app, _mii_app], ids=["vllm", "sglang", "mii"]
)
def test_the_model_listing_is_unchanged(factory):
    resp = _client(factory, []).get("/v1/models", headers={"Origin": "https://example.org"})

    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# soup serve hands each app the address it binds
# ---------------------------------------------------------------------------


def test_serve_vllm_passes_the_bind_address(tmp_path):
    from soup_cli.commands.serve import _serve_vllm

    with (
        patch("soup_cli.utils.vllm.create_vllm_engine", return_value=(MagicMock(), "m")),
        patch("soup_cli.utils.vllm.create_vllm_app", return_value=MagicMock()) as create_app,
        patch("soup_cli.commands.serve._load_serve_tokenizer", return_value=None),
    ):
        _serve_vllm(
            model_path=tmp_path,
            base_model=None,
            is_adapter=False,
            max_tokens_default=16,
            tensor_parallel=1,
            gpu_memory_utilization=0.5,
            host="192.168.1.20",
        )

    assert create_app.call_args.kwargs["host"] == "192.168.1.20"


def test_serve_sglang_passes_the_bind_address(tmp_path):
    from soup_cli.commands.serve import _serve_sglang

    with (
        patch(
            "soup_cli.utils.sglang.create_sglang_runtime",
            return_value=(MagicMock(), "m", None),
        ),
        patch("soup_cli.utils.sglang.create_sglang_app", return_value=MagicMock()) as create_app,
        patch("soup_cli.commands.serve._load_serve_tokenizer", return_value=None),
    ):
        _serve_sglang(
            model_path=tmp_path,
            base_model=None,
            is_adapter=False,
            max_tokens_default=16,
            tensor_parallel=1,
            gpu_memory_utilization=0.5,
            host="192.168.1.20",
        )

    assert create_app.call_args.kwargs["host"] == "192.168.1.20"


TOKEN = "s3cret-token-value"


@pytest.fixture
def model_dir(tmp_path):
    # The command imports uvicorn before it picks a backend, and the tests patch
    # uvicorn.run: without the serve extra there is nothing to run.
    pytest.importorskip("uvicorn")
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def _serve(args):
    from soup_cli.cli import app

    return CliRunner().invoke(app, ["serve", *args])


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_the_command_hands_vllm_and_sglang_the_bind_address(backend, model_dir):
    recorded: dict = {}

    def _record(**kwargs):
        recorded.update(kwargs)
        return MagicMock()

    helper = "_serve_vllm" if backend == "vllm" else "_serve_sglang"
    with (
        patch("soup_cli.utils.vllm.is_vllm_available", return_value=True),
        patch("soup_cli.utils.sglang.check_sglang_available", return_value=True),
        patch(f"soup_cli.commands.serve.{helper}", side_effect=_record),
        patch("uvicorn.run"),
    ):
        result = _serve(
            [
                "--model", str(model_dir), "--backend", backend,
                "--host", "192.168.1.20", "--tool-auth-token", TOKEN,
            ]
        )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert recorded.get("host") == "192.168.1.20", recorded


def test_the_command_hands_mii_the_bind_address(model_dir):
    recorded: dict = {}

    def _record(pipeline, **kwargs):
        recorded.update(kwargs)
        return MagicMock()

    with (
        patch("soup_cli.utils.mii.is_mii_available", return_value=True),
        patch("soup_cli.utils.mii.create_mii_pipeline", return_value=MagicMock()),
        patch("soup_cli.utils.mii.build_mii_app", side_effect=_record),
        patch("soup_cli.commands.serve._load_serve_tokenizer", return_value=None),
        patch("uvicorn.run"),
    ):
        result = _serve(
            [
                "--model", str(model_dir), "--backend", "mii",
                "--host", "192.168.1.20", "--tool-auth-token", TOKEN,
            ]
        )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert recorded.get("host") == "192.168.1.20", recorded
