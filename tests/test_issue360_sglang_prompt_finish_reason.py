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

import json
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


# sglang 0.5.16's ``Runtime.generate`` ends with ``json.dumps(response.json())``
# -- a string, not a dict (#76: every request 500'd with "string indices must be
# integers"). ``decode_sglang_response`` normalises both, and its own unit tests
# cover both, but every app-level test here used to construct the dict shape
# only. So the FastAPI layer -- the thing a real request actually traverses --
# was exercised against just one of the two contracts sglang is known to ship.
RUNTIME_RESPONSE_SHAPES = ("dict", "json_string")


def _mock_runtime(text="hi there", meta_info=None, shape="dict"):
    """Build a Runtime double returning one of sglang's two response shapes."""
    if shape not in RUNTIME_RESPONSE_SHAPES:
        raise ValueError(f"unknown shape {shape!r}")
    runtime = MagicMock()
    payload = {"text": text}
    payload["meta_info"] = meta_info if meta_info is not None else {}
    returned = payload if shape == "dict" else json.dumps(payload)
    runtime.generate = MagicMock(return_value=returned)
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
@pytest.mark.parametrize("shape", RUNTIME_RESPONSE_SHAPES)
class TestSglangReportsFinishReason:
    """Parameterised over both shapes: the #76 string form is what real
    sglang 0.5.16 returns, and the finish_reason contract has to hold through
    the decode step, not only after it."""

    def _post(self, app, body):
        from fastapi.testclient import TestClient

        return TestClient(app).post("/v1/chat/completions", json=body)

    def test_sync_reports_length_when_budget_hit(self, shape):
        from soup_cli.utils.sglang import create_sglang_app

        runtime = _mock_runtime(
            text="a b c", meta_info={"prompt_tokens": 3, "completion_tokens": 8}, shape=shape
        )
        app = create_sglang_app(runtime=runtime, runtime_model_name="m", model_name="m")
        resp = self._post(app, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        assert resp.json()["choices"][0]["finish_reason"] == "length"

    def test_sync_reports_stop_when_under_budget(self, shape):
        from soup_cli.utils.sglang import create_sglang_app

        runtime = _mock_runtime(
            text="a b c", meta_info={"prompt_tokens": 3, "completion_tokens": 3}, shape=shape
        )
        app = create_sglang_app(runtime=runtime, runtime_model_name="m", model_name="m")
        resp = self._post(app, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 128})
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_stream_final_chunk_reports_length(self, shape):
        import json

        from soup_cli.utils.sglang import create_sglang_app

        runtime = _mock_runtime(
            text="a b c", meta_info={"prompt_tokens": 3, "completion_tokens": 8}, shape=shape
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

    def test_control_stream_final_chunk_still_reports_stop(self, shape):
        """Control for the length case above (#360 review item 3).

        Without this, hardcoding the streaming final chunk to ``"length"`` --
        the exact mirror image of the bug being fixed, and one that makes a
        continue-on-length client loop forever -- passes the whole suite.
        Ported from ``test_issue332_vllm_prompt.py``, where vLLM already has it.
        """
        import json

        from soup_cli.utils.sglang import create_sglang_app

        runtime = _mock_runtime(
            text="a b c", meta_info={"prompt_tokens": 3, "completion_tokens": 3}, shape=shape
        )
        app = create_sglang_app(runtime=runtime, runtime_model_name="m", model_name="m")
        resp = self._post(
            app,
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 128, "stream": True},
        )
        reasons = []
        for line in resp.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                payload = json.loads(line[len("data: ") :])
                if "choices" in payload:
                    reasons.append(payload["choices"][0].get("finish_reason"))
        assert reasons[-1] == "stop"


class _TemplatelessTokenizer:
    """A tokenizer that loads fine but ships no chat template.

    Distinct from ``tokenizer=None``: the object exists, so any code that
    branches on truthiness of the tokenizer rather than of its template takes
    the wrong path. #360 asks for the legacy fallback to be pinned for
    template-less models, which is this case, not the missing-tokenizer one.
    """

    chat_template = None

    def apply_chat_template(self, *args, **kwargs):  # pragma: no cover — must not be called
        raise AssertionError(
            "apply_chat_template must not be called on a template-less tokenizer"
        )


@pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
class TestSglangTemplatelessTokenizer:
    """#360 review item 3: pin the fallback for a tokenizer with no template."""

    def _post(self, app, body):
        from fastapi.testclient import TestClient

        return TestClient(app).post("/v1/chat/completions", json=body)

    def test_templateless_tokenizer_uses_the_shared_legacy_fallback(self):
        from soup_cli.utils.sglang import create_sglang_app
        from soup_cli.utils.vllm import build_chat_prompt

        runtime = _mock_runtime()
        app = create_sglang_app(
            runtime=runtime,
            runtime_model_name="m",
            model_name="m",
            tokenizer=_TemplatelessTokenizer(),
        )
        messages = [{"role": "user", "content": "hello"}]
        self._post(app, {"messages": messages})
        sent_prompt = runtime.generate.call_args.args[0]

        # Identical to what the shared builder produces with no tokenizer at
        # all -- the fallback is the shared one, not a third code path.
        assert sent_prompt == build_chat_prompt(messages, None)
        assert not sent_prompt.startswith("<TPL>")


# ---------------------------------------------------------------------------
# Production wiring: what `soup serve --backend sglang` actually executes
# ---------------------------------------------------------------------------
class TestServeSglangWiring:
    """#360 review items 1 and 2.

    Every prompt test above constructs ``create_sglang_app`` directly with a
    hand-supplied tokenizer, so deleting the tokenizer load in ``_serve_sglang``
    -- the one line every real ``soup serve --backend sglang`` runs, and the
    line that makes the whole fix reach users -- passed 112 tests.
    """

    def _serve(self, monkeypatch, tokenizer, capture):
        from pathlib import Path

        import soup_cli.commands.serve as serve_mod
        import soup_cli.utils.sglang as sglang_mod

        monkeypatch.setattr(
            sglang_mod,
            "create_sglang_runtime",
            lambda **kwargs: (MagicMock(), "runtime-model"),
            raising=False,
        )
        monkeypatch.setattr(
            sglang_mod, "create_sglang_app", lambda **kwargs: kwargs, raising=False
        )

        def _fake_loader(**kwargs):
            capture["kwargs"] = kwargs
            return tokenizer

        monkeypatch.setattr(serve_mod, "_load_serve_tokenizer", _fake_loader)

        printed = []
        monkeypatch.setattr(
            serve_mod.console, "print", lambda *args, **kwargs: printed.append(str(args[0]))
        )
        capture["printed"] = printed

        return serve_mod._serve_sglang(
            model_path=Path("/models/custom-code-model"),
            base_model=None,
            is_adapter=False,
            max_tokens_default=256,
            tensor_parallel=1,
            gpu_memory_utilization=0.9,
        )

    def test_tokenizer_load_uses_the_trust_setting_the_runtime_uses(self, monkeypatch):
        """The runtime loads this model with trust_remote_code=True and the
        panel says so. Loading the tokenizer with the default False meant a
        custom-code model produced tokenizer=None and the fix silently did
        nothing -- for exactly the models whose chat template matters most."""
        capture = {}
        self._serve(monkeypatch, _FakeTokenizer(), capture)

        assert capture["kwargs"]["trust_remote_code"] is True

    def test_the_loaded_tokenizer_is_passed_into_the_app(self, monkeypatch):
        """Kills the mutation that deletes the wiring outright."""
        capture = {}
        tokenizer = _FakeTokenizer()
        app_kwargs = self._serve(monkeypatch, tokenizer, capture)

        assert app_kwargs["tokenizer"] is tokenizer

    def test_missing_tokenizer_is_announced(self, monkeypatch):
        """#360 item 2: on vLLM the operator is told; on SGLang the silent
        fallback was invisible."""
        capture = {}
        self._serve(monkeypatch, None, capture)

        assert any("no tokenizer could be loaded" in line for line in capture["printed"])

    def test_templateless_model_is_announced(self, monkeypatch):
        capture = {}
        self._serve(monkeypatch, _TemplatelessTokenizer(), capture)

        assert any("ships no chat template" in line for line in capture["printed"])

    def test_applying_the_models_template_is_announced(self, monkeypatch):
        capture = {}
        self._serve(monkeypatch, _FakeTokenizer(), capture)

        assert any("applying the model's own template" in line for line in capture["printed"])
