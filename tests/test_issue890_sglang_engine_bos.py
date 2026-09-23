"""#890: the SGLang backend no longer lets the engine double the BOS.

#867 fixed the vLLM backend by handing the engine token ids instead of the
rendered string. SGLang's ``Runtime.generate`` posts ``{"text": prompt}`` to
``/generate`` and nothing else, and the server tokenizes that string with its
tokenizer's default ``add_special_tokens=True`` (``TokenizerManager``, no
request or server flag turns it off). For a template that renders
``{{ bos_token }}`` the model received ``[BOS, BOS, ...]``: measured on
SGLang 0.5.9 with ``unsloth/Llama-3.2-1B-Instruct`` as
``meta_info.prompt_tokens`` 37 against the 36 ids
``apply_chat_template(tokenize=True)`` returns.

The same ``/generate`` endpoint takes ``input_ids`` and uses them verbatim
(``TokenizerManager._tokenize_one_request``), so
:func:`generate_with_runtime` posts the ids
:func:`~soup_cli.utils.vllm.build_engine_prompt` encoded for a templated
prompt, and still calls ``Runtime.generate`` with the string for a prompt no
template rendered. Both routes return the ``{"text", "meta_info"}`` dict the
app already parses.

The tokenizer fixtures are the real ``transformers`` fast tokenizers from
``test_issue785_engine_bos``; the runtime is a stand-in that records what it
was handed, and ``httpx.post`` is intercepted so the ids payload is asserted
on without an engine or a GPU.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.test_issue785_engine_bos import (
    _BOS_ID,
    _BOS_TEMPLATE,
    _LEGACY,
    _MESSAGES,
    _hf_prompt_ids,
    _tokenizer,
)

_META = {"prompt_tokens": 36, "completion_tokens": 3, "finish_reason": {"type": "stop"}}


def _engine_would_send(tok, text):
    """What the server builds from a text prompt: ``self.tokenizer(text)``."""
    return tok(text)["input_ids"]


def _runtime(shape="json_string"):
    """A ``Runtime`` double: ``generate`` returns one of sglang's two shapes (#76)."""
    runtime = MagicMock()
    runtime.url = "http://127.0.0.1:30000"
    payload = {"text": "Paris.", "meta_info": dict(_META)}
    runtime.generate = MagicMock(
        return_value=payload if shape == "dict" else json.dumps(payload)
    )
    return runtime


def _http_response(payload=None, status=200):
    """What ``httpx.post`` returns from ``/generate`` for an ids prompt."""
    pytest.importorskip("httpx")
    import httpx

    response = MagicMock()
    response.status_code = status
    response.json.return_value = (
        payload if payload is not None else {"text": "Paris.", "meta_info": dict(_META)}
    )
    if status >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status} from /generate", request=MagicMock(), response=response
        )
    return response


# ============================================================
# The two routes
# ============================================================


class TestGenerateWithRuntime:
    def test_the_fixture_reproduces_the_doubled_bos(self):
        """CONTROL for this whole file: the server's own text route really
        does produce two BOS from the rendered string on this fixture."""
        tok = _tokenizer(_BOS_TEMPLATE)
        text = tok.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=True)

        assert _engine_would_send(tok, text)[:2] == [_BOS_ID, _BOS_ID]

    def test_ids_are_posted_to_generate_and_the_string_route_is_not_used(self):
        from soup_cli.utils.sglang import generate_with_runtime

        runtime = _runtime()
        with patch("httpx.post", return_value=_http_response()) as post:
            generate_with_runtime(runtime, "ignored", [1, 2, 3], {"max_new_tokens": 4})

        post.assert_called_once_with(
            "http://127.0.0.1:30000/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {"max_new_tokens": 4}},
            timeout=300.0,
        )
        runtime.generate.assert_not_called()

    def test_the_ids_route_bounds_the_engine_with_a_timeout(self):
        """The post runs inside the app's event loop, so an unresponsive engine
        must be bounded: ``httpx``'s own 5s default would break long
        generations, so the call passes an explicit finite timeout."""
        from soup_cli.utils.sglang import generate_with_runtime

        with patch("httpx.post", return_value=_http_response()) as post:
            generate_with_runtime(_runtime(), "ignored", [1, 2, 3], {})

        timeout = post.call_args.kwargs["timeout"]
        assert isinstance(timeout, (int, float)) and 0 < timeout < float("inf")

    def test_a_trailing_slash_on_the_runtime_url_does_not_double(self):
        """``runtime.url`` may or may not carry a trailing slash; either way the
        endpoint is ``.../generate``, never ``...//generate``."""
        from soup_cli.utils.sglang import generate_with_runtime

        runtime = _runtime()
        runtime.url = "http://127.0.0.1:30000/"
        with patch("httpx.post", return_value=_http_response()) as post:
            generate_with_runtime(runtime, "ignored", [1, 2, 3], {})

        assert post.call_args.args[0] == "http://127.0.0.1:30000/generate"

    def test_the_ids_route_returns_the_dict_the_string_route_returns(self):
        from soup_cli.utils.sglang import generate_with_runtime

        with patch("httpx.post", return_value=_http_response()):
            from_ids = generate_with_runtime(_runtime(), "ignored", [1, 2], {})
        from_text = generate_with_runtime(_runtime(), "text", None, {})

        assert from_ids == from_text == {"text": "Paris.", "meta_info": _META}

    def test_a_server_error_on_the_ids_route_raises(self):
        """``Runtime.generate`` returns the server's error body as if it were
        a result; the ids route must not do worse than that, so it raises."""
        pytest.importorskip("httpx")
        import httpx

        from soup_cli.utils.sglang import generate_with_runtime

        with patch("httpx.post", return_value=_http_response(status=500)):
            with pytest.raises(httpx.HTTPStatusError):
                generate_with_runtime(_runtime(), "ignored", [1, 2], {})

    @pytest.mark.parametrize("shape", ["dict", "json_string"])
    def test_control_no_ids_calls_runtime_generate_with_the_string(self, shape):
        """CONTROL: ``None`` ids means no template rendered the prompt, and the
        string goes exactly where it always went, in both #76 shapes."""
        from soup_cli.utils.sglang import generate_with_runtime

        runtime = _runtime(shape)
        with patch("httpx.post") as post:
            response = generate_with_runtime(runtime, _LEGACY, None, {"max_new_tokens": 4})

        runtime.generate.assert_called_once_with(_LEGACY, sampling_params={"max_new_tokens": 4})
        post.assert_not_called()
        assert response == {"text": "Paris.", "meta_info": _META}


# ============================================================
# The SGLang backend end to end through its FastAPI app
# ============================================================


def _client(tokenizer, runtime):
    pytest.importorskip("fastapi", reason="the [serve] extra is optional")
    from fastapi.testclient import TestClient

    from soup_cli.utils.sglang import create_sglang_app

    return TestClient(
        create_sglang_app(
            runtime=runtime,
            runtime_model_name="test-model",
            model_name="test-model",
            max_tokens_default=128,
            tokenizer=tokenizer,
        )
    )


def _post(client, stream=False):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": _MESSAGES, "max_tokens": 16, "stream": stream},
    )
    assert response.status_code == 200, response.text
    return response


def _posted_ids(post):
    post.assert_called_once()
    return post.call_args.kwargs["json"]["input_ids"]


class TestSglangBackend:
    @pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
    def test_the_engine_is_handed_token_ids_not_a_string(self, stream):
        """Both routes reach the same generate call; a fix on one only would
        leave the other sending the doubled prompt."""
        tok = _tokenizer(_BOS_TEMPLATE)
        runtime = _runtime()

        with patch("httpx.post", return_value=_http_response()) as post:
            _post(_client(tok, runtime), stream=stream)

        assert post.call_args.args == ("http://127.0.0.1:30000/generate",)
        assert _posted_ids(post) == _hf_prompt_ids(tok, _MESSAGES)
        runtime.generate.assert_not_called()

    @pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
    def test_what_the_engine_receives_carries_exactly_one_bos(self, stream):
        tok = _tokenizer(_BOS_TEMPLATE)

        with patch("httpx.post", return_value=_http_response()) as post:
            _post(_client(tok, _runtime()), stream=stream)

        ids = _posted_ids(post)

        assert ids[0] == _BOS_ID
        assert ids.count(_BOS_ID) == 1

    def test_control_the_string_the_engine_used_to_get_had_two(self):
        """CONTROL: the assertion above is only a finding because the prompt
        this backend sent before really did tokenize to two BOS."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        sent_before = build_chat_prompt(_MESSAGES, tok)

        assert _engine_would_send(tok, sent_before)[:2] == [_BOS_ID, _BOS_ID]

    @pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
    @pytest.mark.parametrize(
        "tokenizer_kind", ["none", "no_template", "broken_template"]
    )
    def test_control_no_template_still_sends_the_legacy_string(self, tokenizer_kind, stream):
        """CONTROL: nothing rendered special tokens, so the engine keeps
        tokenizing the string exactly as it always has."""
        tok = {
            "none": None,
            "no_template": _tokenizer(None),
            "broken_template": _tokenizer("{{ this_is_not_defined.boom() }}"),
        }[tokenizer_kind]
        runtime = _runtime()

        with patch("httpx.post") as post:
            _post(_client(tok, runtime), stream=stream)

        assert runtime.generate.call_args.args == (_LEGACY,)
        post.assert_not_called()

    def test_the_sampling_params_travel_with_the_ids(self):
        tok = _tokenizer(_BOS_TEMPLATE)

        with patch("httpx.post", return_value=_http_response()) as post:
            _post(_client(tok, _runtime()))

        assert post.call_args.kwargs["json"]["sampling_params"] == {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 16,
        }

    def test_the_response_is_parsed_the_same_as_the_string_route(self):
        """The ids route hands back the server's dict directly rather than
        the JSON string ``Runtime.generate`` returns (#76); the app must read
        text, usage and finish_reason from it exactly as before."""
        tok = _tokenizer(_BOS_TEMPLATE)
        meta = {"prompt_tokens": 36, "completion_tokens": 16, "finish_reason": {"type": "length"}}

        with patch(
            "httpx.post",
            return_value=_http_response({"text": "Paris, and more", "meta_info": meta}),
        ):
            body = _post(_client(tok, _runtime())).json()

        assert body["choices"][0]["message"]["content"] == "Paris, and more"
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["usage"] == {"prompt_tokens": 36, "completion_tokens": 16, "total_tokens": 52}

    def test_the_streamed_chunks_are_built_from_the_ids_route_response(self):
        tok = _tokenizer(_BOS_TEMPLATE)

        with patch("httpx.post", return_value=_http_response()):
            text = _post(_client(tok, _runtime()), stream=True).text

        chunks = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Paris."
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert text.endswith("data: [DONE]\n\n")

    def test_a_server_error_on_the_ids_route_is_a_500_not_a_hang_or_a_leak(self):
        tok = _tokenizer(_BOS_TEMPLATE)
        client = _client(tok, _runtime())

        with patch("httpx.post", return_value=_http_response(status=503)):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": _MESSAGES, "max_tokens": 16},
            )

        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error"}
