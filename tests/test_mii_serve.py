"""Tests for the DeepSpeed-MII backend's OpenAI-compatible HTTP layer (#606).

`mii.py` had no dedicated test file, which is how its `/v1/chat/completions`
handler kept two defects the vLLM and transformers backends had already fixed
for the same failure classes:

  1. it hand-rolled a `"System: ...\\nUser: ...\\nAssistant:"` prompt instead of
     calling `build_chat_prompt`, so a chat-tuned model saw a format it never
     trained on (the #332 failure class), and
  2. it hardcoded `finish_reason: "stop"`, so a client could not tell a
     truncated answer from a completed one (the #333 failure class).

No GPU or `deepspeed-mii` install is needed: `build_mii_app` accepts any
callable `pipeline(prompts, max_new_tokens=...) -> [Response]`, so a stub
exercises exactly the HTTP layer a real MII engine would hit.
"""

from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


# ============================================================
# Stub pipeline / tokenizer
# ============================================================
class _Response:
    """Stand-in for an MII ``Response``."""

    def __init__(self, generated_text, finish_reason=None, token_ids=None):
        self.generated_text = generated_text
        if finish_reason is not None:
            self.finish_reason = finish_reason
        if token_ids is not None:
            self.token_ids = token_ids


def _recording_pipeline(response=None):
    """Return ``(pipeline, captured)``; ``captured["prompt"]`` is what it saw."""
    captured = {}

    def pipeline(prompts, max_new_tokens=None, **kwargs):
        captured["prompt"] = prompts[0]
        captured["max_new_tokens"] = max_new_tokens
        captured.update(kwargs)
        if response is not None:
            return [response]
        return [_Response("ok")]

    return pipeline, captured


def _template_tokenizer(rendered="<|im_start|>rendered<|im_end|>"):
    """A tokenizer that reports a chat template and applies it."""
    tokenizer = MagicMock()
    tokenizer.chat_template = "{{ messages }}"
    tokenizer.apply_chat_template.return_value = rendered
    return tokenizer


def _client(pipeline, tokenizer=None, **kwargs):
    from soup_cli.utils.mii import build_mii_app

    app = build_mii_app(pipeline, model_name="test-model", tokenizer=tokenizer, **kwargs)
    return TestClient(app)


MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "2+2?"},
]


def _post(client, **overrides):
    payload = {"model": "test-model", "messages": MESSAGES}
    payload.update(overrides)
    return client.post("/v1/chat/completions", json=payload)


# ============================================================
# Defect 1 — the model's chat template (#332 failure class)
# ============================================================
class TestMiiChatTemplate:
    """The MII backend must render prompts through the shared builder."""

    def test_applies_the_models_chat_template(self):
        """A tokenizer with a template must be used, not the legacy format."""
        pipeline, captured = _recording_pipeline()
        tokenizer = _template_tokenizer()

        resp = _post(_client(pipeline, tokenizer=tokenizer))

        assert resp.status_code == 200
        assert captured["prompt"] == "<|im_start|>rendered<|im_end|>"
        # The regression being guarded: the hand-rolled format must be gone.
        assert "User: 2+2?" not in captured["prompt"]
        tokenizer.apply_chat_template.assert_called_once()

    def test_template_is_asked_for_a_generation_prompt(self):
        """Without add_generation_prompt the model is not cued to reply."""
        pipeline, _ = _recording_pipeline()
        tokenizer = _template_tokenizer()

        _post(_client(pipeline, tokenizer=tokenizer))

        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["tokenize"] is False

    def test_falls_back_to_legacy_prompt_without_a_tokenizer(self):
        """No tokenizer must degrade to the documented legacy format."""
        pipeline, captured = _recording_pipeline()

        resp = _post(_client(pipeline, tokenizer=None))

        assert resp.status_code == 200
        assert captured["prompt"] == "System: You are terse.\nUser: 2+2?\nAssistant:"

    def test_falls_back_when_the_model_ships_no_template(self):
        """A tokenizer without a chat_template is the same fallback."""
        pipeline, captured = _recording_pipeline()
        tokenizer = MagicMock()
        tokenizer.chat_template = None

        _post(_client(pipeline, tokenizer=tokenizer))

        assert captured["prompt"] == "System: You are terse.\nUser: 2+2?\nAssistant:"

    def test_a_broken_template_does_not_500(self):
        """A template that raises must fall back, not fail the request."""
        pipeline, captured = _recording_pipeline()
        tokenizer = _template_tokenizer()
        tokenizer.apply_chat_template.side_effect = ValueError("bad template")

        resp = _post(_client(pipeline, tokenizer=tokenizer))

        assert resp.status_code == 200
        assert captured["prompt"] == "System: You are terse.\nUser: 2+2?\nAssistant:"


# ============================================================
# Defect 2 — finish_reason (#333 failure class)
# ============================================================
class TestMiiFinishReason:
    """finish_reason must reflect what actually happened."""

    def test_engine_reported_length_is_preserved(self):
        """The exact regression: 'length' must not be flattened to 'stop'."""
        pipeline, _ = _recording_pipeline(_Response("x" * 4, finish_reason="length"))

        resp = _post(_client(pipeline), max_tokens=4)

        assert resp.json()["choices"][0]["finish_reason"] == "length"

    def test_engine_reported_stop_is_preserved(self):
        pipeline, _ = _recording_pipeline(_Response("done", finish_reason="stop"))

        resp = _post(_client(pipeline), max_tokens=64)

        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_full_budget_is_derived_as_length_without_a_reported_reason(self):
        """An output that used its whole budget was truncated, not stopped."""
        pipeline, _ = _recording_pipeline(_Response("xxxx", token_ids=[1, 2, 3, 4]))

        resp = _post(_client(pipeline), max_tokens=4)

        assert resp.json()["choices"][0]["finish_reason"] == "length"

    def test_short_output_is_stop(self):
        pipeline, _ = _recording_pipeline(_Response("hi", token_ids=[1, 2]))

        resp = _post(_client(pipeline), max_tokens=64)

        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_unmapped_engine_reason_is_not_leaked_to_clients(self):
        """A non-OpenAI value must be normalised, not passed through."""
        pipeline, _ = _recording_pipeline(_Response("x", finish_reason="abort"))

        resp = _post(_client(pipeline), max_tokens=64)

        assert resp.json()["choices"][0]["finish_reason"] in {"stop", "length"}

    def test_default_max_tokens_is_used_for_the_derivation(self):
        """max_tokens omitted must fall back to max_tokens_default."""
        pipeline, captured = _recording_pipeline(_Response("xx", token_ids=[1, 2]))

        resp = _post(_client(pipeline, max_tokens_default=2))

        assert captured["max_new_tokens"] == 2
        assert resp.json()["choices"][0]["finish_reason"] == "length"


# ============================================================
# Existing behaviour that must not regress
# ============================================================
class TestMiiAppUnchangedBehaviour:
    """The rest of the handler's contract is untouched by #606."""

    def test_response_envelope_is_still_openai_shaped(self):
        pipeline, _ = _recording_pipeline(_Response("4"))

        body = _post(_client(pipeline)).json()

        assert body["object"] == "chat.completion"
        assert body["model"] == "test-model"
        assert body["choices"][0]["index"] == 0
        assert body["choices"][0]["message"] == {"role": "assistant", "content": "4"}
        assert body["id"].startswith("chatcmpl-")

    def test_streaming_is_still_rejected(self):
        pipeline, _ = _recording_pipeline()

        assert _post(_client(pipeline), stream=True).status_code == 400

    @pytest.mark.parametrize("bad", [0, -1, 16385])
    def test_max_tokens_bounds_still_enforced(self, bad):
        pipeline, _ = _recording_pipeline()

        assert _post(_client(pipeline), max_tokens=bad).status_code == 400

    def test_empty_response_list_still_500s(self):
        def pipeline(prompts, max_new_tokens=None, **kwargs):
            return []

        assert _post(_client(pipeline)).status_code == 500

    def test_pipeline_exception_still_500s(self):
        def pipeline(prompts, max_new_tokens=None, **kwargs):
            raise RuntimeError("engine died")

        assert _post(_client(pipeline)).status_code == 500

    def test_models_endpoint_still_lists_the_model(self):
        pipeline, _ = _recording_pipeline()

        body = _client(pipeline).get("/v1/models").json()

        assert body["data"][0]["id"] == "test-model"

    def test_sampling_params_are_forwarded(self):
        pipeline, captured = _recording_pipeline()

        _post(_client(pipeline), temperature=0.1, top_p=0.5)

        assert captured["temperature"] == 0.1
        assert captured["top_p"] == 0.5


# ============================================================
# The call site (#608 review) — mutation (d)
# ============================================================
class TestServeMiiCallSite:
    """`soup serve --backend mii` must LOAD a tokenizer and PASS it.

    Every other test in this file constructs the app directly with a
    hand-supplied tokenizer, which leaves the one line a user actually
    executes uncovered: deleting `tokenizer=mii_tokenizer` from the
    `build_mii_app(...)` call in `serve.py` makes every real
    `soup serve --backend mii` fall back to the legacy prompt while the
    whole suite stays green.

    These drive the real command through its CLI and assert on the call
    kwargs, so that deletion fails by name.
    """

    def _invoke(self, tmp_path, *, tokenizer):
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        captured: dict[str, object] = {}

        def _fake_build_mii_app(pipeline, **kwargs):
            captured.update(kwargs)
            captured["called"] = True
            return MagicMock()

        with (
            patch("soup_cli.utils.mii.is_mii_available", return_value=True),
            patch("soup_cli.utils.mii.create_mii_pipeline", return_value=MagicMock()),
            patch("soup_cli.utils.mii.build_mii_app", side_effect=_fake_build_mii_app),
            patch(
                "soup_cli.commands.serve._load_serve_tokenizer", return_value=tokenizer
            ) as loader,
            patch("uvicorn.run"),
        ):
            runner = CliRunner()
            result = runner.invoke(
                app, ["serve", "--model", str(model_dir), "--backend", "mii"]
            )
        return result, captured, loader

    def test_serve_mii_passes_the_loaded_tokenizer_to_build_mii_app(self, tmp_path):
        """Mutation (d): dropping `tokenizer=` from the call must fail here."""
        sentinel = object()
        result, captured, loader = self._invoke(tmp_path, tokenizer=sentinel)

        assert captured.get("called"), f"build_mii_app was never reached: {result.output}"
        assert "tokenizer" in captured, (
            "serve() called build_mii_app without a tokenizer kwarg - every real "
            "`soup serve --backend mii` would fall back to the legacy prompt"
        )
        assert captured["tokenizer"] is sentinel, (
            "serve() passed something other than the tokenizer it loaded"
        )
        loader.assert_called_once()

    def test_serve_mii_still_starts_when_no_tokenizer_can_be_loaded(self, tmp_path):
        """The documented degrade: None is passed through, not an exception.

        `build_chat_prompt` falls back to the legacy format for a None
        tokenizer, so the server must still come up.
        """
        result, captured, _ = self._invoke(tmp_path, tokenizer=None)

        assert captured.get("called"), f"build_mii_app was never reached: {result.output}"
        assert captured["tokenizer"] is None
