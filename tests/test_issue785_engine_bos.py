"""#785 part 2: the backends that let the ENGINE tokenize the rendered prompt.

#782 fixed every path that tokenizes in process: they render the chat template
and then encode with ``add_special_tokens=False``. It deliberately left the
backends that hand an inference engine a prompt STRING alone, because nobody had
the hardware to check what those engines' own tokenizers do with it.

Measured on an RTX 3060 (WSL2) with ``unsloth/Llama-3.2-1B-Instruct``, whose
Llama-3 template renders ``{{- bos_token }}`` and whose ``bos_token_id`` is
128000. One request, ``messages=[{"role": "user", "content": "Hi"}]``:

=================  ==========================  ===============================
engine             what Soup sent              engine ``prompt_token_ids[:2]``
=================  ==========================  ===============================
vLLM 0.29.0        the rendered string         ``[128000, 128000]``  (37 ids)
vLLM 0.29.0        ``{"prompt_token_ids":..}`` ``[128000, 128006]``  (36 ids)
SGLang 0.5.9 (*)   the rendered string         ``[128000, 128000]``  (37 ids)
MII 0.3.3 (*)      the rendered string         ``[128000, 128000]``  (37 ids)
=================  ==========================  ===============================

(*) through the engine's own tokenizer path (SGLang's ``TokenizerManager``
call, MII's ``HFTokenizer.encode``) on the real tokenizer; the GPU engine
itself could not be started on that box. The vLLM rows are from a running
engine's ``RequestOutput.prompt_token_ids``.

36 ids with ``[128000, 128006]`` is exactly what
``apply_chat_template(tokenize=True)`` returns, and exactly the prefix
``data/loss_mask.py`` trains on. So the string prompt really did put a second
BOS in front of the one the template had already rendered, and sending the ids
really does remove it. This PR fixes the vLLM backend; SGLang and MII are
measured here and split out, per the issue.

The tokenizers below are real ``transformers`` fast tokenizers built offline
from an in-memory vocab, so ``apply_chat_template`` is the genuine Jinja
renderer and the BOS comes from a genuine ``tokenizers`` post-processor. The
engines are stand-ins that record what they were handed; the GPU measurement
above is what says a real engine tokenizes a string this way; these tests pin
that Soup stops handing it one.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

_SPECIALS = ["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|end|>"]
_WORDS = ["You", "are", "terse", ".", "What", "is", "the", "capital", "of", "France", "?",
          "System", "User", "Assistant", ":"]
_BOS_ID = _SPECIALS.index("<s>")

# The vendor shape this issue is about (Llama-3, Gemma, Mistral): the template
# renders BOS itself, and the tokenizer's post-processor would add another.
_BOS_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

_SYSTEM = {"role": "system", "content": "You are terse."}
_USER = {"role": "user", "content": "What is the capital of France?"}
_MESSAGES = [_SYSTEM, _USER]

_LEGACY = "System: You are terse.\nUser: What is the capital of France?\nAssistant:"


def _tokenizer(chat_template, *, post_processor="bos"):
    """A real fast tokenizer whose post-processor prepends ``<s>``."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, processors

    vocab = {token: index for index, token in enumerate(_SPECIALS + _WORDS)}
    backend = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(_SPECIALS)
    if post_processor == "bos":
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", _BOS_ID)]
        )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = chat_template
    return tok


def _engine_would_send(tok, text):
    """What an engine that tokenizes the string itself builds from ``text``.

    This is the pre-fix encoding: the engine's tokenizer with its own default
    ``add_special_tokens=True``, which is what vLLM was measured doing.
    """
    return tok(text)["input_ids"]


def _hf_prompt_ids(tok, messages):
    """HF's own one-step encoding, the reference the fix has to reproduce."""
    return list(
        tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
    )


# ============================================================
# The shared builder
# ============================================================


class TestBuildEnginePrompt:
    def test_the_fixture_reproduces_the_doubled_bos(self):
        """CONTROL for this whole file: on this fixture an engine that tokenizes
        the rendered string really does produce two BOS. Without it, a
        post-processor that silently added nothing would let every assertion
        below pass vacuously."""
        tok = _tokenizer(_BOS_TEMPLATE)
        text = tok.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=True)

        assert _engine_would_send(tok, text)[:2] == [_BOS_ID, _BOS_ID]

    def test_a_rendered_prompt_yields_ids_with_exactly_one_bos(self):
        from soup_cli.utils.vllm import build_engine_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        _, ids = build_engine_prompt(_MESSAGES, tok)

        assert ids[0] == _BOS_ID
        assert ids.count(_BOS_ID) == 1

    def test_the_ids_equal_hf_apply_chat_template_with_tokenize_true(self):
        """The same invariant #782 pinned for the in-process paths: what the
        engine receives must be what HF's one-step encoding produces."""
        from soup_cli.utils.vllm import build_engine_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        _, ids = build_engine_prompt(_MESSAGES, tok)

        assert ids == _hf_prompt_ids(tok, _MESSAGES)

    def test_the_text_is_still_exactly_what_build_chat_prompt_renders(self):
        """The fix changes how the prompt is ENCODED, never how it is rendered.
        #332's template choice has to survive it untouched."""
        from soup_cli.utils.vllm import build_chat_prompt, build_engine_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        text, _ = build_engine_prompt(_MESSAGES, tok)

        assert text == build_chat_prompt(_MESSAGES, tok)

    def test_a_soup_preset_is_prompted_with_no_bos_at_all(self):
        """Soup's own ``data.chat_template`` presets render no BOS, and training
        adds none, so an adapter trained under one never saw a BOS. The engine
        was adding one anyway.

        This is the case that separates the invariant from "drop a duplicate
        BOS": that rule would leave this prompt with a BOS training never had."""
        from soup_cli.data.chat_templates import apply_chat_template_override
        from soup_cli.utils.vllm import build_engine_prompt

        tok = _tokenizer(None)
        apply_chat_template_override(tok, "llama3")
        text = tok.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=True)
        assert not text.startswith("<s>")
        assert _engine_would_send(tok, text)[0] == _BOS_ID  # the defect, on the preset

        _, ids = build_engine_prompt(_MESSAGES, tok)

        assert _BOS_ID not in ids
        assert ids == _hf_prompt_ids(tok, _MESSAGES)

    def test_control_a_tokenizer_that_adds_nothing_gets_the_same_ids_as_before(self):
        """CONTROL: with no post-processor there was never a defect here, so
        the fix must not move a single id."""
        from soup_cli.utils.vllm import build_engine_prompt

        tok = _tokenizer(_BOS_TEMPLATE, post_processor=None)
        text = tok.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=True)

        _, ids = build_engine_prompt(_MESSAGES, tok)

        assert ids == _engine_would_send(tok, text)

    @pytest.mark.parametrize(
        ("tokenizer_kind", "reason"),
        [("none", "no tokenizer could be loaded"), ("no_template", "the model ships none")],
        ids=["no-tokenizer", "no-template"],
    )
    def test_control_no_template_sends_the_legacy_string_and_no_ids(
        self, tokenizer_kind, reason
    ):
        """CONTROL: the legacy role-prefixed fallback carries no special tokens
        of its own, so the engine must go on adding its own. ``None`` ids is how
        each backend knows to send the string, exactly as it always did."""
        from soup_cli.utils.vllm import build_engine_prompt

        tok = None if tokenizer_kind == "none" else _tokenizer(None)

        text, ids = build_engine_prompt(_MESSAGES, tok)

        assert text == _LEGACY, reason
        assert ids is None, reason

    def test_a_broken_template_falls_back_to_the_legacy_string(self):
        """The flag has to mean "the template rendered this text", not "the
        tokenizer has a template". The fallback text never went through the
        template, so the engine's own BOS is the only one it should get."""
        from soup_cli.utils.vllm import build_engine_prompt

        tok = _tokenizer("{{ this_is_not_defined.boom() }}")

        text, ids = build_engine_prompt(_MESSAGES, tok)

        assert text == _LEGACY
        assert ids is None


# ============================================================
# The vLLM backend, the engine this was measured on
# ============================================================


class _FakeOutput:
    def __init__(self, text=" Paris.", token_ids=(1, 2, 3)):
        self.text = text
        self.token_ids = list(token_ids)
        self.finish_reason = "stop"


class _FakeRequestOutput:
    def __init__(self, output, prompt_token_ids=(1, 2, 3)):
        self.outputs = [output]
        self.prompt_token_ids = list(prompt_token_ids)


def _vllm_client(tokenizer, capture):
    pytest.importorskip("fastapi", reason="the [serve] extra is optional")
    from fastapi.testclient import TestClient

    engine = MagicMock()

    def _generate(prompt, sampling_params, request_id, **kwargs):
        capture["prompt"] = prompt

        async def _gen():
            yield _FakeRequestOutput(_FakeOutput())

        return _gen()

    engine.generate = _generate

    vllm_stub = MagicMock()
    vllm_stub.SamplingParams = MagicMock()
    with patch.dict(
        sys.modules,
        {"vllm": vllm_stub, "vllm.lora": MagicMock(), "vllm.lora.request": MagicMock()},
    ):
        from soup_cli.utils.vllm import create_vllm_app

        app = create_vllm_app(
            engine=engine,
            engine_model_name="test-model",
            model_name="test-model",
            max_tokens_default=128,
            tokenizer=tokenizer,
        )
    return TestClient(app)


def _post(client, stream=False):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": _MESSAGES,
            "max_tokens": 16,
            "stream": stream,
        },
    )
    assert response.status_code == 200, response.text
    return response


class TestVllmBackend:
    @pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
    def test_the_engine_is_handed_token_ids_not_a_string(self, stream):
        """Both routes reach the same ``engine.generate``; a fix on one only
        would leave the other sending the doubled prompt."""
        tok = _tokenizer(_BOS_TEMPLATE)
        capture = {}

        _post(_vllm_client(tok, capture), stream=stream)

        assert capture["prompt"] == {"prompt_token_ids": _hf_prompt_ids(tok, _MESSAGES)}

    @pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
    def test_what_the_engine_receives_carries_exactly_one_bos(self, stream):
        tok = _tokenizer(_BOS_TEMPLATE)
        capture = {}

        _post(_vllm_client(tok, capture), stream=stream)

        ids = capture["prompt"]["prompt_token_ids"]

        assert ids[0] == _BOS_ID
        assert ids.count(_BOS_ID) == 1

    def test_control_the_string_the_engine_used_to_get_had_two(self):
        """CONTROL: the assertion above is only a finding because the prompt
        this backend sent before really did tokenize to two BOS."""
        tok = _tokenizer(_BOS_TEMPLATE)
        from soup_cli.utils.vllm import build_chat_prompt

        sent_before = build_chat_prompt(_MESSAGES, tok)

        assert _engine_would_send(tok, sent_before)[:2] == [_BOS_ID, _BOS_ID]

    @pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
    def test_control_a_model_with_no_template_still_gets_the_legacy_string(self, stream):
        """CONTROL: nothing rendered special tokens, so the engine keeps
        tokenizing the string exactly as it always has."""
        capture = {}

        _post(_vllm_client(_tokenizer(None), capture), stream=stream)

        assert capture["prompt"] == _LEGACY

    def test_usage_still_reports_the_prompt_length_the_engine_reports(self):
        """The engine keeps reporting ``prompt_token_ids`` for a tokens prompt,
        so the token accounting in the response must not change shape."""
        tok = _tokenizer(_BOS_TEMPLATE)

        body = _post(_vllm_client(tok, {})).json()

        assert body["usage"]["prompt_tokens"] == 3  # len(_FakeRequestOutput ids)
