"""#360 — port the two vLLM prompt/finish_reason fixes to the SGLang backend.

v0.73.0 fixed two defects in the vLLM serve backend and left the identical pair
standing in ``utils/sglang.py``:

1. The prompt was hand-rolled as ``"User: …\nAssistant:"`` instead of applying
   the model's own chat template — the model saw a format it was never trained
   on. The shared ``build_chat_prompt`` (used by the transformers and vLLM
   backends) applies the template with a warned legacy fallback for
   template-less models; SGLang should use it, not a third copy.
2. ``finish_reason`` was hardcoded ``"stop"``, so a client doing
   continue-on-length could not tell a completed answer from one truncated at
   ``max_tokens``.

These are CPU-verifiable (prompt string + finish_reason mapping); a live SGLang
runtime on Linux is a separate follow-up per the issue.
"""

from unittest.mock import MagicMock

import pytest


def _has_fastapi() -> bool:
    try:
        import fastapi  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# resolve_sglang_finish_reason — the mapping, mirrors vLLM's resolve_finish_reason
# ---------------------------------------------------------------------------
class TestResolveSglangFinishReason:
    def _fn(self):
        from soup_cli.utils.sglang import resolve_sglang_finish_reason

        return resolve_sglang_finish_reason

    def test_reported_dict_type_length(self):
        # SGLang's modern shape: meta_info["finish_reason"] == {"type": "length"}.
        assert self._fn()({"finish_reason": {"type": "length"}}, 128) == "length"

    def test_reported_dict_type_stop(self):
        assert self._fn()({"finish_reason": {"type": "stop"}}, 128) == "stop"

    def test_reported_bare_string_older_shape(self):
        # Older SGLang reported a bare string — accept it too (a control, so a
        # dict-only fix cannot silently break previously-working versions).
        assert self._fn()({"finish_reason": "length"}, 128) == "length"

    def test_derives_length_from_token_budget(self):
        # No reported reason: a generation that spent the whole budget was
        # truncated, not naturally stopped.
        assert self._fn()({"completion_tokens": 128}, 128) == "length"

    def test_derives_stop_when_under_budget(self):
        assert self._fn()({"completion_tokens": 4}, 128) == "stop"

    def test_unmapped_reason_falls_through_to_token_count(self):
        # "abort" is not an OpenAI finish_reason — do not leak it; derive.
        meta = {"finish_reason": {"type": "abort"}, "completion_tokens": 4}
        assert self._fn()(meta, 128) == "stop"

    def test_missing_meta_info_is_stop(self):
        assert self._fn()(None, 128) == "stop"
        assert self._fn()({}, None) == "stop"


# ---------------------------------------------------------------------------
# The app uses the shared build_chat_prompt (not the hand-rolled "User:/Assistant:")
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    chat_template = "A-REAL-TEMPLATE"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = " ".join(f"{m['role']}={m['content']}" for m in messages)
        return f"<TPL>{rendered}<GEN>"


def _mock_runtime(text="hi there", meta_info=None):
    runtime = MagicMock()
    payload = {"text": text}
    payload["meta_info"] = meta_info if meta_info is not None else {}
    runtime.generate = MagicMock(return_value=payload)
    return runtime


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
class TestSglangUsesSharedPromptBuilder:
    def _post(self, app, body):
        from fastapi.testclient import TestClient

        return TestClient(app).post("/v1/chat/completions", json=body)

    def test_prompt_uses_chat_template_when_tokenizer_has_one(self):
        from soup_cli.utils.sglang import create_sglang_app

        runtime = _mock_runtime()
        app = create_sglang_app(
            runtime=runtime,
            runtime_model_name="m",
            model_name="m",
            tokenizer=_FakeTokenizer(),
        )
        self._post(app, {"messages": [{"role": "user", "content": "hello"}]})
        sent_prompt = runtime.generate.call_args.args[0]
        assert sent_prompt.startswith("<TPL>")
        assert "user=hello" in sent_prompt
        # The hand-rolled format must be gone.
        assert "User: hello" not in sent_prompt

    def test_prompt_matches_shared_builder_legacy_fallback_without_tokenizer(self):
        from soup_cli.utils.sglang import create_sglang_app
        from soup_cli.utils.vllm import build_chat_prompt

        runtime = _mock_runtime()
        app = create_sglang_app(
            runtime=runtime, runtime_model_name="m", model_name="m", tokenizer=None
        )
        messages = [{"role": "user", "content": "hello"}]
        self._post(app, {"messages": messages})
        sent_prompt = runtime.generate.call_args.args[0]
        # SGLang now delegates to the shared builder — identical output.
        assert sent_prompt == build_chat_prompt(messages, None)


# ---------------------------------------------------------------------------
# finish_reason is reported truthfully on both the sync and streaming paths
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
class TestSglangReportsFinishReason:
    def _post(self, app, body):
        from fastapi.testclient import TestClient

        return TestClient(app).post("/v1/chat/completions", json=body)

    def test_sync_reports_length_when_budget_hit(self):
        from soup_cli.utils.sglang import create_sglang_app

        runtime = _mock_runtime(
            text="a b c", meta_info={"prompt_tokens": 3, "completion_tokens": 8}
        )
        app = create_sglang_app(runtime=runtime, runtime_model_name="m", model_name="m")
        resp = self._post(app, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        assert resp.json()["choices"][0]["finish_reason"] == "length"

    def test_sync_reports_stop_when_under_budget(self):
        from soup_cli.utils.sglang import create_sglang_app

        runtime = _mock_runtime(
            text="a b c", meta_info={"prompt_tokens": 3, "completion_tokens": 3}
        )
        app = create_sglang_app(runtime=runtime, runtime_model_name="m", model_name="m")
        resp = self._post(app, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 128})
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_stream_final_chunk_reports_length(self):
        import json

        from soup_cli.utils.sglang import create_sglang_app

        runtime = _mock_runtime(
            text="a b c", meta_info={"prompt_tokens": 3, "completion_tokens": 8}
        )
        app = create_sglang_app(runtime=runtime, runtime_model_name="m", model_name="m")
        resp = self._post(
            app,
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "stream": True},
        )
        reasons = []
        for line in resp.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                payload = json.loads(line[len("data: ") :])
                if "choices" in payload:
                    reasons.append(payload["choices"][0].get("finish_reason"))
        assert reasons[-1] == "length"
