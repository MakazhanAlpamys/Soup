"""Real offline Metal model/adapter and HTTP contracts; no replaced native boundary."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from benchmarks.harness.mlx_serve_fixture import save_fixture  # noqa: E402


def raw_reference(path: Path, messages: list[dict], max_tokens: int = 4) -> dict:
    """Compare HTTP against the engine, independently of Soup's streaming wrapper."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from soup_cli.utils.mlx_generation import load_native_model
    from soup_cli.utils.vllm import _render_chat_prompt

    model, tokenizer = load_native_model(str(path), require_dense=True)
    rendered, templated = _render_chat_prompt(messages, tokenizer, fallback_on_error=False)
    prompt = tokenizer.encode(rendered, add_special_tokens=not templated)
    responses = list(
        stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0)
        )
    )
    last = responses[-1]
    return dict(
        text="".join(item.text for item in responses),
        prompt_tokens=len(prompt),
        completion_tokens=last.generation_tokens,
        finish_reason=last.finish_reason,
        first_token=int(responses[0].token),
    )


def native_app(path: Path):
    from soup_cli.commands.serve import _create_app
    from soup_cli.utils.mlx_serve import NativeServerRuntime

    runtime = NativeServerRuntime(str(path))
    return _create_app(None, None, "mlx", "fixture", 4, native_runtime=runtime), runtime


def assert_response(response: dict, expected: dict) -> None:
    assert response["choices"][0]["message"]["content"] == expected["text"]
    assert response["choices"][0]["finish_reason"] == expected["finish_reason"]
    assert response["usage"] == dict(
        prompt_tokens=expected["prompt_tokens"],
        completion_tokens=expected["completion_tokens"],
        total_tokens=expected["prompt_tokens"] + expected["completion_tokens"],
    )


def test_real_dense_and_trained_adapter_match_raw_engine_across_requests(tmp_path):
    fixture = save_fixture(tmp_path)
    prompts = [[{"role": "user", "content": content}] for content in ["red green", "t6 red"]]
    adapted = [raw_reference(fixture["adapter"], prompt) for prompt in prompts]
    base = [raw_reference(fixture["base"], prompt) for prompt in prompts]
    assert any(a["text"] != b["text"] for a, b in zip(adapted, base, strict=True)), (
        "fixture must catch silently ignored adapters"
    )
    for path, expected in [(fixture["base"], base), (fixture["adapter"], adapted)]:
        app, runtime = native_app(path)
        with TestClient(app) as client:
            assert client.get("/health").json()["status"] == "ok"
            for index in [0, 1, 0]:
                body = dict(model="fixture", messages=prompts[index], temperature=0, max_tokens=4)
                response = client.post("/v1/chat/completions", json=body)
                assert response.status_code == 200, response.text
                assert_response(response.json(), expected[index])
                streamed = client.post("/v1/chat/completions", json={**body, "stream": True})
                frames = [
                    json.loads(line[6:])
                    for line in streamed.text.splitlines()
                    if line.startswith("data: {")
                ]
                assert (
                    "".join(f["choices"][0]["delta"].get("content", "") for f in frames)
                    == (expected[index]["text"])
                )
                assert frames[-1]["usage"] == response.json()["usage"]
                assert frames[-1]["choices"][0]["finish_reason"] == expected[index]["finish_reason"]
                assert streamed.text.endswith("data: [DONE]\n\n")
        assert not runtime._thread.is_alive()


def test_real_eos_length_stop_and_input_token_cap(tmp_path, monkeypatch):
    from soup_cli.utils import mlx_serve

    fixture = save_fixture(tmp_path)
    messages = [{"role": "user", "content": "red green"}]
    reference = raw_reference(fixture["base"], messages, 1)
    assert reference["finish_reason"] == "length"
    eos_base = tmp_path / "eos-base"
    shutil.copytree(fixture["base"], eos_base)
    config = json.loads((eos_base / "config.json").read_text())
    config["eos_token_id"] = reference["first_token"]
    (eos_base / "config.json").write_text(json.dumps(config))
    app, _ = native_app(eos_base)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=dict(messages=messages, temperature=0))
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["finish_reason"] == "stop"
        assert response.json()["usage"]["completion_tokens"] == 1
        assert response.json()["choices"][0]["message"]["content"] == ""

    app, _ = native_app(fixture["base"])
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=dict(
                messages=messages,
                temperature=0,
                stop=reference["text"],
            ),
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["finish_reason"] == "stop"
        assert response.json()["choices"][0]["message"]["content"] == ""
        monkeypatch.setattr(mlx_serve, "MAX_PROMPT_TOKENS", 1)
        response = client.post("/v1/chat/completions", json=dict(messages=messages, temperature=0))
        assert response.status_code == 422


def test_native_server_refuses_real_quantized_base_and_corrupt_adapter(tmp_path):
    from mlx_lm import load
    from mlx_lm.utils import quantize_model, save_config, save_model

    from soup_cli.utils.mlx_generation import load_native_model

    fixture = save_fixture(tmp_path)
    model, _, config = load(str(fixture["base"]), return_config=True)
    model, config = quantize_model(model, config, 32, 4)
    quantized = tmp_path / "quantized"
    shutil.copytree(fixture["base"], quantized)
    save_model(quantized, model)
    save_config(config, quantized / "config.json")
    # Existing batch inference retains its qualified 4-bit support.
    load_native_model(str(quantized))
    app, runtime = native_app(quantized)
    with pytest.raises(RuntimeError, match="dense checkpoints only"):
        with TestClient(app):
            pytest.fail("quantized serving was not qualified")
    assert not runtime.ready and not runtime._thread.is_alive()
    (fixture["adapter"] / "adapters.safetensors").write_bytes(b"corrupt")
    app, runtime = native_app(fixture["adapter"])
    with pytest.raises(RuntimeError, match="Cannot read native MLX adapter"):
        with TestClient(app):
            pytest.fail("corrupt adapter must not be ignored")
    assert not runtime.ready and not runtime._thread.is_alive()
