"""#891: the DeepSpeed-MII backend no longer lets the engine double the BOS.

#867 fixed the vLLM backend by handing the engine token ids instead of the
rendered string. MII's pipeline takes strings only (``MIIPipeline.__call__``),
and encodes each one with ``HFTokenizer.encode``, which hardcodes
``tokenizer.encode(input, return_tensors="pt")`` and so the tokenizer's default
``add_special_tokens=True``. For a template that renders ``{{ bos_token }}`` the
model received ``[BOS, BOS, ...]``: measured on MII 0.3.3 with
``unsloth/Llama-3.2-1B-Instruct`` as ``prompt_length`` 37 against the 36 ids
``apply_chat_template(tokenize=True)`` returns.

The seam on MII 0.3.3 (``mii/modeling/tokenizers.py``): ``mii.pipeline`` takes
``tokenizer=``, and ``load_tokenizer`` wraps whatever object it gets in a fresh
``HFTokenizer`` that calls ``obj.encode(input, return_tensors="pt").flatten()``
on the prompt object the pipeline was handed, untouched. So Soup hands MII
:func:`create_mii_tokenizer` around the served tokenizer, and hands the
pipeline a template-rendered prompt as a :class:`TokenizedPrompt`: still a
``str``, so the pipeline's contract holds, but carrying the ids
:func:`build_engine_prompt` built with no tokenizer special tokens, which that
encode returns verbatim. The legacy fallback prompt stays a plain ``str`` and is
encoded with the exact call MII always made.

``_StubHFTokenizer`` below is that one ``HFTokenizer.encode`` line copied from
MII 0.3.3, so the double wrap MII really performs is exercised here without a
GPU or ``deepspeed-mii`` install. The tokenizer fixtures are the real
``transformers`` fast tokenizers from ``test_issue785_engine_bos``.
"""

import logging
import sys
from types import ModuleType
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


def _mii_would_send(tok, text):
    """MII 0.3.3's ``HFTokenizer.encode``, verbatim: the pre-fix encoding."""
    pytest.importorskip("torch")
    return tok.encode(text, return_tensors="pt").flatten().tolist()


class _StubHFTokenizer:
    """``mii.modeling.tokenizers.HFTokenizer`` as MII 0.3.3 ships it.

    Only what ``load_tokenizer`` and the pipeline touch: the constructor keeps
    a non-str object as-is, ``encode`` is the hardcoded line, ``vocab_size``
    and ``eos_token_id`` read through the object.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def encode(self, input):
        return self.tokenizer.encode(input, return_tensors="pt").flatten()

    def convert_tokens_to_ids(self, input):
        return self.tokenizer.convert_tokens_to_ids(input)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


def _stub_mii():
    """A ``mii`` package whose tokenizer module is the 0.3.3 seam above."""
    mii = MagicMock()
    modeling = ModuleType("mii.modeling")
    tokenizers = ModuleType("mii.modeling.tokenizers")
    tokenizers.HFTokenizer = _StubHFTokenizer
    tokenizers.MIITokenizerWrapper = object
    return {"mii": mii, "mii.modeling": modeling, "mii.modeling.tokenizers": tokenizers}


def _tokenized(tok, messages=_MESSAGES):
    """The prompt the app hands the pipeline for a templated request."""
    from soup_cli.utils.mii import TokenizedPrompt
    from soup_cli.utils.vllm import build_engine_prompt

    text, ids = build_engine_prompt(messages, tok)
    assert ids is not None
    return TokenizedPrompt(text, ids)


def _as_mii_loads_it(tok):
    """What ``load_tokenizer`` makes of ``create_mii_tokenizer(tok)`` on 0.3.3."""
    pytest.importorskip("torch")
    with patch.dict(sys.modules, _stub_mii()):
        from soup_cli.utils.mii import create_mii_tokenizer

        return _StubHFTokenizer(create_mii_tokenizer(tok))


# ============================================================
# The encoder MII is given
# ============================================================


class TestEncodeMiiPrompt:
    def test_the_fixture_reproduces_the_doubled_bos_through_miis_encode_line(self):
        """CONTROL for this whole file: MII's own encode really does produce
        two BOS from the rendered string on this fixture."""
        tok = _tokenizer(_BOS_TEMPLATE)
        text = tok.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=True)

        assert _mii_would_send(tok, text)[:2] == [_BOS_ID, _BOS_ID]

    def test_a_tokenized_prompt_encodes_to_the_ids_it_carries(self):
        from soup_cli.utils.mii import TokenizedPrompt, encode_mii_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        assert encode_mii_prompt(tok, TokenizedPrompt("anything", [7, 8, 9])) == [7, 8, 9]

    def test_a_templated_prompt_encodes_with_exactly_one_bos(self):
        from soup_cli.utils.mii import encode_mii_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        ids = encode_mii_prompt(tok, _tokenized(tok))

        assert ids[0] == _BOS_ID
        assert ids.count(_BOS_ID) == 1

    def test_the_ids_equal_hf_apply_chat_template_with_tokenize_true(self):
        from soup_cli.utils.mii import encode_mii_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        assert encode_mii_prompt(tok, _tokenized(tok)) == _hf_prompt_ids(tok, _MESSAGES)

    def test_the_prompt_is_still_the_string_build_chat_prompt_renders(self):
        """The fix changes what the engine ENCODES, never what the template
        rendered; the string MII sees is unchanged."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        assert _tokenized(tok) == build_chat_prompt(_MESSAGES, tok)

    def test_a_soup_preset_is_prompted_with_no_bos_at_all(self):
        """Same case #867 pinned: a template that renders no BOS means the
        engine gets none, not "one instead of two"."""
        from soup_cli.data.chat_templates import apply_chat_template_override
        from soup_cli.utils.mii import encode_mii_prompt

        tok = _tokenizer(None)
        apply_chat_template_override(tok, "llama3")
        text = tok.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=True)
        assert _mii_would_send(tok, text)[0] == _BOS_ID  # the defect, on the preset

        ids = encode_mii_prompt(tok, _tokenized(tok))

        assert _BOS_ID not in ids
        assert ids == _hf_prompt_ids(tok, _MESSAGES)

    def test_control_a_plain_string_encodes_exactly_as_mii_always_did(self):
        """CONTROL: the legacy prompt is a plain str and carries no special
        tokens of its own, so MII's own BOS must still be added to it, by the
        very call MII makes."""
        from soup_cli.utils.mii import encode_mii_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        encoded = encode_mii_prompt(tok, _LEGACY, return_tensors="pt")

        assert encoded.flatten().tolist() == _mii_would_send(tok, _LEGACY)
        assert encoded.flatten().tolist()[0] == _BOS_ID
        assert encode_mii_prompt(tok, _LEGACY) == tok.encode(_LEGACY)


class TestTokenizedPromptCopyHazard:
    """The fix rides a ``str`` subclass that carries ``token_ids`` in
    ``__slots__``; any str operation that builds a NEW str drops them and the
    prompt silently reverts to the doubled BOS with no exception. These pin the
    sharp edge and the ``_check_prompt_length`` guard that catches its one
    observable symptom at runtime (#891 review)."""

    @pytest.mark.parametrize(
        "copy",
        [
            lambda p: p.strip(),
            lambda p: f"{p}",
            lambda p: p + "",
            lambda p: p[:],
            lambda p: "".join([p]),
        ],
        ids=["strip", "f-string", "concat", "slice", "join"],
    )
    def test_a_copied_prompt_is_a_plain_str_that_lost_its_ids(self, copy):
        from soup_cli.utils.mii import TokenizedPrompt

        copied = copy(TokenizedPrompt("rendered", [7, 8, 9]))

        assert not isinstance(copied, TokenizedPrompt)
        assert not hasattr(copied, "token_ids")

    def test_encoding_a_copied_prompt_reverts_to_the_doubled_bos(self):
        """The consequence a copy causes: with its ids gone, the prompt is
        re-encoded by the tokenizer, which adds its own BOS again — the exact
        defect this PR removes, back in silence."""
        from soup_cli.utils.mii import encode_mii_prompt

        tok = _tokenizer(_BOS_TEMPLATE)
        copied = f"{_tokenized(tok)}"  # content-identical, but a plain str

        assert encode_mii_prompt(tok, copied)[:2] == [_BOS_ID, _BOS_ID]

    def test_check_prompt_length_is_the_runtime_guard_for_that_reversion(self, caplog):
        """So the reversion cannot pass unseen: the engine reports how many ids
        it ran on, and ``_check_prompt_length`` warns when that is not the count
        Soup encoded (one more == the engine re-added its BOS)."""
        from soup_cli.utils.mii import _check_prompt_length

        soup_ids = [1, 2, 3]

        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.mii"):
            _check_prompt_length(MagicMock(prompt_length=len(soup_ids) + 1), soup_ids)
        assert any("#785" in r.message for r in caplog.records)

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.mii"):
            _check_prompt_length(MagicMock(prompt_length=len(soup_ids)), soup_ids)
        assert not [r for r in caplog.records if r.name == "soup_cli.utils.mii"]


# ============================================================
# The wrapper, through the double wrap MII 0.3.3 performs
# ============================================================


class TestCreateMiiTokenizer:
    def test_the_seam_wraps_it_again_and_still_gets_one_bos(self):
        """``load_tokenizer`` builds ``HFTokenizer(our_wrapper)`` and calls
        ``our_wrapper.encode(input, return_tensors="pt").flatten()``. A wrapper
        that only overrode ``encode(input)`` would raise TypeError here."""
        tok = _tokenizer(_BOS_TEMPLATE)

        ids = _as_mii_loads_it(tok).encode(_tokenized(tok)).tolist()

        assert ids == _hf_prompt_ids(tok, _MESSAGES)
        assert ids.count(_BOS_ID) == 1

    def test_control_a_plain_string_through_the_seam_is_unchanged(self):
        tok = _tokenizer(_BOS_TEMPLATE)

        ids = _as_mii_loads_it(tok).encode(_LEGACY).tolist()

        assert ids == _mii_would_send(tok, _LEGACY)

    def test_encode_returns_the_flat_int64_tensor_the_pipeline_batches(self):
        """``_put_request`` hands the result straight to ``make_request``;
        a 2-D tensor or a list would break the ragged batch."""
        import torch

        tok = _tokenizer(_BOS_TEMPLATE)

        for prompt in (_tokenized(tok), _LEGACY):
            encoded = _as_mii_loads_it(tok).encode(prompt)
            assert isinstance(encoded, torch.Tensor)
            assert encoded.dim() == 1
            assert encoded.dtype == torch.int64

    def test_as_the_wrapper_itself_encode_is_a_flat_tensor_too(self):
        """Should a later MII stop re-wrapping ``ModelConfig.tokenizer``, the
        object is still a working ``MIITokenizerWrapper``: ``encode(input)``
        with no kwargs returns the flat tensor ``_put_request`` expects."""
        import torch

        tok = _tokenizer(_BOS_TEMPLATE)
        with patch.dict(sys.modules, _stub_mii()):
            from soup_cli.utils.mii import create_mii_tokenizer

            wrapper = create_mii_tokenizer(tok)

        for prompt in (_tokenized(tok), _LEGACY):
            encoded = wrapper.encode(prompt)
            assert isinstance(encoded, torch.Tensor)
            assert encoded.dim() == 1
        assert wrapper.encode(_tokenized(tok)).tolist() == _hf_prompt_ids(tok, _MESSAGES)
        assert wrapper.encode(_LEGACY).tolist() == _mii_would_send(tok, _LEGACY)

    def test_everything_else_mii_reads_comes_from_the_real_tokenizer(self):
        """The wrapper stands in for the HF tokenizer, so the attributes MII
        reads on it (vocab size, EOS, decode, stop-token lookup) must be the
        real tokenizer's, or generation would stop on the wrong id."""
        tok = _tokenizer(_BOS_TEMPLATE)

        wrapped = _as_mii_loads_it(tok)

        assert wrapped.vocab_size == len(tok)
        assert wrapped.eos_token_id == tok.eos_token_id
        assert wrapped.convert_tokens_to_ids("</s>") == tok.convert_tokens_to_ids("</s>")
        assert wrapped.decode([_BOS_ID]) == tok.decode([_BOS_ID])
        assert wrapped.tokenizer.bos_token == tok.bos_token  # any other attribute
        assert wrapped.tokenizer.pad_token == tok.eos_token  # what HFTokenizer sets

    def test_the_wrapper_is_truthy_so_mii_does_not_replace_it_with_the_path(self):
        """``ModelConfig`` falls back to loading from ``model_name_or_path``
        when ``not values.get("tokenizer")``."""
        with patch.dict(sys.modules, _stub_mii()):
            from soup_cli.utils.mii import create_mii_tokenizer

            assert create_mii_tokenizer(_tokenizer(_BOS_TEMPLATE))


class TestCreateMiiPipeline:
    def test_the_wrapped_tokenizer_is_passed_to_mii(self):
        stubs = _stub_mii()
        tok = _tokenizer(_BOS_TEMPLATE)
        with patch.dict(sys.modules, stubs):
            from soup_cli.utils.mii import create_mii_pipeline

            create_mii_pipeline(model_path="/models/llama", tokenizer=tok)

        kwargs = stubs["mii"].pipeline.call_args.kwargs
        assert kwargs["tokenizer"].tokenizer is tok
        assert isinstance(kwargs["tokenizer"], _StubHFTokenizer)  # the config's type

    def test_control_without_a_tokenizer_mii_loads_its_own_as_before(self):
        stubs = _stub_mii()
        with patch.dict(sys.modules, stubs):
            from soup_cli.utils.mii import create_mii_pipeline

            create_mii_pipeline(model_path="/models/llama")

        assert "tokenizer" not in stubs["mii"].pipeline.call_args.kwargs


# ============================================================
# The HTTP layer: what the pipeline is handed
# ============================================================


def _recording_pipeline(prompt_length=None):
    """A pipeline that records the prompt it was handed, as MII would get it."""
    pytest.importorskip("fastapi", reason="the [serve] extra is optional")
    captured = {}

    def pipeline(prompts, **kwargs):
        captured["prompt"] = prompts[0]
        response = MagicMock()
        response.generated_text = "Paris."
        response.finish_reason = "stop"
        if prompt_length is not None:
            response.prompt_length = prompt_length
        else:
            del response.prompt_length
        return [response]

    return pipeline, captured


def _post(tokenizer, pipeline):
    from fastapi.testclient import TestClient

    from soup_cli.utils.mii import build_mii_app

    client = TestClient(build_mii_app(pipeline, model_name="test-model", tokenizer=tokenizer))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": _MESSAGES, "max_tokens": 16},
    )
    assert response.status_code == 200, response.text
    return response


class TestMiiBackend:
    def test_a_templated_prompt_reaches_the_pipeline_carrying_its_ids(self):
        from soup_cli.utils.mii import TokenizedPrompt

        tok = _tokenizer(_BOS_TEMPLATE)
        pipeline, captured = _recording_pipeline()

        _post(tok, pipeline)

        assert isinstance(captured["prompt"], TokenizedPrompt)
        assert captured["prompt"].token_ids == _hf_prompt_ids(tok, _MESSAGES)
        assert captured["prompt"] == tok.apply_chat_template(
            _MESSAGES, tokenize=False, add_generation_prompt=True
        )

    def test_what_the_engine_encodes_carries_exactly_one_bos(self):
        """End to end through the seam: the marked prompt the app sends, encoded
        by the tokenizer MII would have been built with."""
        tok = _tokenizer(_BOS_TEMPLATE)
        pipeline, captured = _recording_pipeline()

        _post(tok, pipeline)
        ids = _as_mii_loads_it(tok).encode(captured["prompt"]).tolist()

        assert ids == _hf_prompt_ids(tok, _MESSAGES)
        assert ids.count(_BOS_ID) == 1

    def test_control_the_string_the_engine_used_to_get_had_two(self):
        """CONTROL: the assertion above is only a finding because the prompt
        this backend sent before really did encode to two BOS."""
        from soup_cli.utils.vllm import build_chat_prompt

        tok = _tokenizer(_BOS_TEMPLATE)

        sent_before = build_chat_prompt(_MESSAGES, tok)

        assert _mii_would_send(tok, sent_before)[:2] == [_BOS_ID, _BOS_ID]

    @pytest.mark.parametrize(
        "tokenizer_kind", ["none", "no_template", "broken_template"]
    )
    def test_control_no_template_sends_a_plain_legacy_string(self, tokenizer_kind):
        """CONTROL: nothing rendered special tokens, so the prompt must reach
        MII as a plain str and be tokenized exactly as it always was."""
        from soup_cli.utils.mii import TokenizedPrompt

        tok = {
            "none": None,
            "no_template": _tokenizer(None),
            "broken_template": _tokenizer("{{ this_is_not_defined.boom() }}"),
        }[tokenizer_kind]
        pipeline, captured = _recording_pipeline()

        _post(tok, pipeline)

        assert captured["prompt"] == _LEGACY
        assert not isinstance(captured["prompt"], TokenizedPrompt)
        if tok is not None:
            encoded = _as_mii_loads_it(tok).encode(captured["prompt"]).tolist()
            assert encoded == _mii_would_send(tok, _LEGACY)

    def test_a_prompt_length_the_encoder_did_not_produce_is_reported(self, caplog):
        """The engine says how many ids it ran on. One more than Soup encoded
        means the pipeline is not using the wrapped tokenizer and the BOS is
        doubled again; that must not pass in silence."""
        tok = _tokenizer(_BOS_TEMPLATE)
        expected = len(_hf_prompt_ids(tok, _MESSAGES))
        pipeline, _ = _recording_pipeline(prompt_length=expected + 1)

        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.mii"):
            _post(tok, pipeline)

        assert any("#785" in record.message for record in caplog.records)

    @pytest.mark.parametrize("case", ["matching", "no_prompt_length", "legacy"])
    def test_control_a_consistent_prompt_length_is_quiet(self, case, caplog):
        tok = _tokenizer(_BOS_TEMPLATE)
        expected = len(_hf_prompt_ids(tok, _MESSAGES))
        if case == "matching":
            pipeline, _ = _recording_pipeline(prompt_length=expected)
        elif case == "no_prompt_length":
            pipeline, _ = _recording_pipeline()
        else:
            tok = _tokenizer(None)
            pipeline, _ = _recording_pipeline(prompt_length=expected + 1)

        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.mii"):
            _post(tok, pipeline)

        assert not [r for r in caplog.records if r.name == "soup_cli.utils.mii"]
