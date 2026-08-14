"""#332 / #333 — the vLLM serve backend's prompt, finish_reason, /metrics and
``--max-model-len``.

#332: ``utils/vllm.py`` hand-rolled a ``"User: ...\\nAssistant:"`` prompt while
the transformers backend applied the model's own chat template. On
Llama-3.1-8B + LoRA that produced a run-on loop that burned the whole token
budget (reproduced on an H100 — see the report attached to the issue).

The fix is a single shared builder, so the two backends cannot drift again.
Every test below that asserts the template path is paired with a CONTROL that
pins the legacy fallback, because "uses apply_chat_template" is only meaningful
if the no-template case still produces the old string.

#333: three smaller defects from the same run — ``finish_reason`` hardcoded
``"stop"``, ``--dashboard`` silently no-opping (no ``/metrics`` route at all),
and no ``--max-model-len`` lever although ``create_vllm_engine`` already
accepted one.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_CHATML = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

_MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is the capital of France?"},
]

_LEGACY = (
    "System: You are terse.\n"
    "User: What is the capital of France?\n"
    "Assistant:"
)


def _tokenizer(chat_template):
    """A real ``transformers`` tokenizer — so ``apply_chat_template`` is the
    genuine Jinja renderer, not a mock that would agree with anything.

    Built offline from an in-memory vocab: no network, no model download.
    """
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")

    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab={"<unk>": 0}, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>"
    )
    tok.chat_template = chat_template
    return tok


class _Msg:
    """Stand-in for the pydantic ``ChatMessage`` the vLLM app builds."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


# ============================================================
# #332 — the prompt builder
# ============================================================


class TestBuildChatPrompt:
    """The shared builder used by both serve backends."""

    def test_prompt_equals_apply_chat_template(self):
        """The whole issue: the built prompt must BE the template render."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer(_CHATML)
        expected = tok.apply_chat_template(
            _MESSAGES, tokenize=False, add_generation_prompt=True
        )

        assert build_chat_prompt(_MESSAGES, tok) == expected

    def test_the_templated_prompt_is_not_the_legacy_one(self):
        """Control for the test above: without it, a builder that ignored the
        tokenizer would still pass if the template happened to render the same
        text. Pins that the two really differ on this fixture."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer(_CHATML)
        built = build_chat_prompt(_MESSAGES, tok)

        assert "<|im_start|>" in built
        assert built != _LEGACY
        assert "User: What is the capital of France?" not in built

    def test_control_tokenizer_without_template_uses_legacy_format(self):
        """CONTROL — a model that ships no chat template must still be served,
        byte-for-byte as before the fix."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer(None)

        assert build_chat_prompt(_MESSAGES, tok) == _LEGACY

    def test_control_no_tokenizer_at_all_uses_legacy_format(self):
        """CONTROL — the tokenizer is optional (it can fail to load); the
        builder must degrade, never raise."""
        from soup_cli.utils.vllm import build_chat_prompt

        assert build_chat_prompt(_MESSAGES, None) == _LEGACY

    def test_pydantic_style_messages_and_dicts_agree(self):
        """The vLLM app passes objects, the transformers app passes dicts —
        one builder, one answer."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer(_CHATML)
        objs = [_Msg(m["role"], m["content"]) for m in _MESSAGES]

        assert build_chat_prompt(objs, tok) == build_chat_prompt(_MESSAGES, tok)

    def test_extra_message_keys_reach_the_template(self):
        """The transformers backend already served tool-call / multimodal rows
        whose dicts carry more than role+content. The shared builder must not
        quietly drop those keys."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer(
            "{% for m in messages %}{{ m['role'] }}:{{ m.get('name', '-') }}"
            "{% endfor %}"
        )
        msgs = [{"role": "tool", "content": "42", "name": "calculator"}]

        assert build_chat_prompt(msgs, tok) == "tool:calculator"

    def test_unknown_roles_are_kept_for_the_template(self):
        """A ``tool`` message must reach the template, not be dropped."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer(_CHATML)
        msgs = [{"role": "tool", "content": "42"}]

        assert "<|im_start|>tool" in build_chat_prompt(msgs, tok)

    def test_a_broken_template_does_not_kill_the_request(self):
        """A template that raises falls back to the legacy format rather than
        500-ing every request on that model."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer("{{ this_is_not_defined.boom() }}")

        assert build_chat_prompt(_MESSAGES, tok) == _LEGACY


class TestBothBackendsShareOneBuilder:
    """Acceptance #1: vLLM and transformers must produce the SAME string."""

    def test_transformers_backend_calls_the_shared_builder(self):
        src = Path("src/soup_cli/commands/serve.py").read_text(encoding="utf-8")
        assert "build_chat_prompt(" in src, (
            "the transformers backend must use the shared builder"
        )

    def test_no_second_hand_rolled_prompt_remains_in_the_serve_backends(self):
        """The literal that produced the run-on loop must exist in exactly one
        place — inside the shared fallback."""
        serve_src = Path("src/soup_cli/commands/serve.py").read_text(encoding="utf-8")
        vllm_src = Path("src/soup_cli/utils/vllm.py").read_text(encoding="utf-8")

        assert 'f"User: {content}"' not in serve_src
        assert vllm_src.count('f"User: {content}"') == 1


# ============================================================
# vLLM app-level behaviour (mocked engine — no GPU needed)
# ============================================================


class _FakeOutput:
    def __init__(self, text, token_ids, finish_reason=None):
        self.text = text
        self.token_ids = token_ids
        if finish_reason is not None:
            self.finish_reason = finish_reason


class _FakeRequestOutput:
    def __init__(self, output, prompt_token_ids=(1, 2, 3)):
        self.outputs = [output]
        self.prompt_token_ids = list(prompt_token_ids)


def _fake_engine(output, capture):
    """An engine whose ``generate`` records the prompt it was handed."""

    engine = MagicMock()

    def _generate(prompt, sampling_params, request_id, **kwargs):
        capture["prompt"] = prompt
        capture["sampling_params"] = sampling_params

        async def _gen():
            yield _FakeRequestOutput(output)

        return _gen()

    engine.generate = _generate
    return engine


def _build_app(*, tokenizer=None, output=None, capture=None, **kwargs):
    pytest.importorskip("fastapi")
    vllm_stub = MagicMock()
    vllm_stub.SamplingParams = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "vllm": vllm_stub,
            "vllm.lora": MagicMock(),
            "vllm.lora.request": MagicMock(),
        },
    ):
        from soup_cli.utils.vllm import create_vllm_app

        built = create_vllm_app(
            engine=_fake_engine(output, capture if capture is not None else {}),
            engine_model_name="test-model",
            model_name="test-model",
            max_tokens_default=128,
            tokenizer=tokenizer,
            **kwargs,
        )
    return built


class TestVllmAppPrompt:
    """The route that actually reached the H100."""

    def _post(self, tokenizer, max_tokens=16):
        pytest.importorskip("fastapi", reason="the [serve] extra is optional")
        from fastapi.testclient import TestClient

        capture = {}
        app = _build_app(
            tokenizer=tokenizer,
            output=_FakeOutput(" Paris.", [1, 2, 3]),
            capture=capture,
        )
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": _MESSAGES,
                "max_tokens": max_tokens,
            },
        )
        assert resp.status_code == 200, resp.text
        return capture, resp.json()

    def test_engine_receives_the_templated_prompt(self):
        tok = _tokenizer(_CHATML)
        capture, _ = self._post(tok)

        assert capture["prompt"] == tok.apply_chat_template(
            _MESSAGES, tokenize=False, add_generation_prompt=True
        )

    def test_control_engine_receives_legacy_prompt_without_a_template(self):
        capture, _ = self._post(_tokenizer(None))

        assert capture["prompt"] == _LEGACY


# ============================================================
# #333.1 — finish_reason
# ============================================================


class TestResolveFinishReason:
    def test_engine_reported_length_is_reported_as_length(self):
        from soup_cli.utils.vllm import resolve_finish_reason

        out = _FakeOutput("x", [1] * 64, finish_reason="length")
        assert resolve_finish_reason(out, 64) == "length"

    def test_engine_reported_stop_is_reported_as_stop(self):
        from soup_cli.utils.vllm import resolve_finish_reason

        out = _FakeOutput("x", [1, 2], finish_reason="stop")
        assert resolve_finish_reason(out, 64) == "stop"

    def test_derived_from_token_count_when_the_engine_says_nothing(self):
        """Older vLLM builds leave ``finish_reason`` unset mid-stream."""
        from soup_cli.utils.vllm import resolve_finish_reason

        assert resolve_finish_reason(_FakeOutput("x", [1] * 64), 64) == "length"

    def test_control_short_output_with_no_engine_reason_is_stop(self):
        from soup_cli.utils.vllm import resolve_finish_reason

        assert resolve_finish_reason(_FakeOutput("x", [1, 2]), 64) == "stop"

    def test_unknown_engine_reason_is_normalised_to_stop(self):
        """``abort`` is not an OpenAI finish_reason; never leak it verbatim."""
        from soup_cli.utils.vllm import resolve_finish_reason

        out = _FakeOutput("x", [1, 2], finish_reason="abort")
        assert resolve_finish_reason(out, 64) == "stop"


class TestVllmAppFinishReason:
    """The observed defect: ``"stop"`` with completion_tokens == max_tokens."""

    def _post(self, output, max_tokens):
        pytest.importorskip("fastapi", reason="the [serve] extra is optional")
        from fastapi.testclient import TestClient

        app = _build_app(
            tokenizer=_tokenizer(_CHATML), output=output, capture={}
        )
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": _MESSAGES,
                "max_tokens": max_tokens,
            },
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    def test_length_truncation_reports_length(self):
        body = self._post(_FakeOutput("essay", [1] * 64, "length"), 64)

        assert body["usage"]["completion_tokens"] == 64
        assert body["choices"][0]["finish_reason"] == "length"

    def test_control_natural_stop_still_reports_stop(self):
        body = self._post(_FakeOutput("Paris.", [1, 2, 3], "stop"), 64)

        assert body["choices"][0]["finish_reason"] == "stop"

    def test_anthropic_route_maps_length_to_max_tokens(self):
        pytest.importorskip("fastapi", reason="the [serve] extra is optional")
        from fastapi.testclient import TestClient

        app = _build_app(
            tokenizer=_tokenizer(_CHATML),
            output=_FakeOutput("essay", [1] * 8, "length"),
            capture={},
        )
        resp = TestClient(app).post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["stop_reason"] == "max_tokens"

    def test_control_anthropic_route_still_maps_stop_to_end_turn(self):
        pytest.importorskip("fastapi", reason="the [serve] extra is optional")
        from fastapi.testclient import TestClient

        app = _build_app(
            tokenizer=_tokenizer(_CHATML),
            output=_FakeOutput("Paris.", [1, 2], "stop"),
            capture={},
        )
        resp = TestClient(app).post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["stop_reason"] == "end_turn"


class TestVllmStreamFinishReason:
    """The SSE path hardcoded ``"stop"`` in its final chunk too."""

    def _final_chunk(self, output, max_tokens):
        import json

        pytest.importorskip("fastapi", reason="the [serve] extra is optional")
        from fastapi.testclient import TestClient

        app = _build_app(
            tokenizer=_tokenizer(_CHATML), output=output, capture={}
        )
        with TestClient(app).stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": _MESSAGES,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            frames = [
                json.loads(line[len("data: "):])
                for line in resp.iter_lines()
                if line.startswith("data: ") and not line.endswith("[DONE]")
            ]
        return frames[-1]

    def test_stream_final_chunk_reports_length(self):
        chunk = self._final_chunk(_FakeOutput("essay", [1] * 12, "length"), 12)

        assert chunk["choices"][0]["finish_reason"] == "length"

    def test_control_stream_final_chunk_still_reports_stop(self):
        chunk = self._final_chunk(_FakeOutput("Paris.", [1, 2], "stop"), 12)

        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_streamed_requests_are_counted_by_metrics(self):
        pytest.importorskip("fastapi", reason="the [serve] extra is optional")
        from fastapi.testclient import TestClient

        app = _build_app(
            tokenizer=_tokenizer(_CHATML),
            output=_FakeOutput("Paris.", [1, 2, 3], "stop"),
            capture={},
        )
        client = TestClient(app)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": _MESSAGES,
                "max_tokens": 8,
                "stream": True,
            },
        ) as resp:
            list(resp.iter_lines())

        snapshot = client.get("/metrics").json()
        assert snapshot["requests_total"] == 1
        assert snapshot["tokens_generated_total"] == 3


# ============================================================
# #333.2 — /metrics
# ============================================================


class TestVllmMetrics:
    def test_metrics_route_exists(self):
        app = _build_app(
            tokenizer=None, output=_FakeOutput("x", [1]), capture={}
        )

        assert "/metrics" in [r.path for r in app.routes if hasattr(r, "path")]

    def test_metrics_counts_a_served_request(self):
        pytest.importorskip("fastapi", reason="the [serve] extra is optional")
        from fastapi.testclient import TestClient

        app = _build_app(
            tokenizer=_tokenizer(_CHATML),
            output=_FakeOutput("Paris.", [1, 2, 3], "stop"),
            capture={},
        )
        client = TestClient(app)

        before = client.get("/metrics")
        assert before.status_code == 200
        assert before.json()["requests_total"] == 0

        client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": _MESSAGES, "max_tokens": 8},
        )

        after = client.get("/metrics").json()
        assert after["requests_total"] == 1
        assert after["tokens_generated_total"] == 3
        assert after["latency_samples"] == 1

    def test_dashboard_intent_is_visible_on_the_app(self):
        app = _build_app(
            tokenizer=None,
            output=_FakeOutput("x", [1]),
            capture={},
            enable_dashboard=True,
        )

        assert app.state.enable_dashboard is True


class TestDashboardBackendWarning:
    """Acceptance: ``--dashboard`` must not silently no-op."""

    def test_no_warning_for_backends_that_serve_metrics(self):
        from soup_cli.commands.serve import _dashboard_warning

        assert _dashboard_warning("transformers") is None
        assert _dashboard_warning("vllm") is None

    def test_sglang_warns_that_metrics_is_not_served(self):
        from soup_cli.commands.serve import _dashboard_warning

        warning = _dashboard_warning("sglang")
        assert warning is not None
        assert "sglang" in warning
        assert "/metrics" in warning

    def test_serve_emits_the_warning(self):
        src = Path("src/soup_cli/commands/serve.py").read_text(encoding="utf-8")
        assert "_dashboard_warning(backend)" in src


# ============================================================
# #333.3 — --max-model-len
# ============================================================


class TestMaxModelLenFlag:
    def test_serve_exposes_the_flag(self):
        import typer.main

        from soup_cli.cli import app as cli_app

        group = typer.main.get_command(cli_app)
        command = group.commands["serve"]
        opts = {opt for param in command.params for opt in param.opts}

        assert "--max-model-len" in opts

    def test_serve_vllm_forwards_max_model_len(self, tmp_path):
        pytest.importorskip("fastapi")
        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch(
            "soup_cli.utils.vllm.create_vllm_engine",
            return_value=(MagicMock(), "base-model"),
        ) as engine_factory, patch(
            "soup_cli.utils.vllm.create_vllm_app", return_value=MagicMock()
        ):
            from soup_cli.commands.serve import _serve_vllm

            _serve_vllm(
                model_path=model_path,
                base_model="base-model",
                is_adapter=False,
                max_tokens_default=512,
                tensor_parallel=1,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
            )

        assert engine_factory.call_args.kwargs["max_model_len"] == 4096

    def test_control_default_is_none(self, tmp_path):
        pytest.importorskip("fastapi")
        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch(
            "soup_cli.utils.vllm.create_vllm_engine",
            return_value=(MagicMock(), "base-model"),
        ) as engine_factory, patch(
            "soup_cli.utils.vllm.create_vllm_app", return_value=MagicMock()
        ):
            from soup_cli.commands.serve import _serve_vllm

            _serve_vllm(
                model_path=model_path,
                base_model="base-model",
                is_adapter=False,
                max_tokens_default=512,
                tensor_parallel=1,
                gpu_memory_utilization=0.9,
            )

        assert engine_factory.call_args.kwargs["max_model_len"] is None
