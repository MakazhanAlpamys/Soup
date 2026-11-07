"""Native streaming text/count contracts with an explicitly replaced engine."""

import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def engine(monkeypatch):
    from soup_cli.utils import vllm

    state = SimpleNamespace(responses=[], calls=[], closed=False, sampler=None)
    native = ModuleType("mlx_lm")
    sampling = ModuleType("mlx_lm.sample_utils")

    def generate(*args, **kwargs):
        state.calls.append(kwargs)
        try:
            yield from state.responses
        finally:
            state.closed = True

    def sampler(**kwargs):
        state.sampler = kwargs
        return kwargs

    native.stream_generate = generate
    sampling.make_sampler = sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", native)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sampling)
    monkeypatch.setattr(vllm, "_render_chat_prompt", lambda *a, **k: ("rendered", True))
    state.tokenizer = SimpleNamespace(encode=lambda text, **kw: [1, 2, 3])
    return state


def _response(text, tokens, reason=None):
    return SimpleNamespace(text=text, generation_tokens=tokens, finish_reason=reason)


@pytest.mark.parametrize(
    "pieces,stops,expected,reason",
    [
        (["alpha<", "ST", "OP>suffix"], ("<STOP>",), "alpha", "stop"),
        (["é中E", "NDtail"], ("END", "STOP"), "é中", "stop"),
        (["abc", "d"], ("abcdef",), "abcd", "length"),
        (["hello", "世界"], (), "hello世界", "length"),
        ([" x ", " y "], (), " x  y ", "length"),
        (["ab", "a"], ("aba", "ab"), "", "stop"),
    ],
)
def test_stops_span_chunks_without_leaking_prefixes(engine, pieces, stops, expected, reason):
    from soup_cli.utils.mlx_generation import stream_native

    engine.responses = [
        _response(text, i + 1, "length" if i == len(pieces) - 1 else None)
        for i, text in enumerate(pieces)
    ]
    chunks = list(
        stream_native(
            None, engine.tokenizer, [], stop=stops, max_tokens=8, top_p=0.4, temperature=0.6
        )
    )
    assert "".join(chunk.text for chunk in chunks) == expected
    assert chunks[-1].finish_reason == reason
    assert sum(chunk.finish_reason is not None for chunk in chunks) == 1
    assert all(chunk.prompt_tokens == 3 for chunk in chunks)
    assert engine.sampler == {"temp": 0.6, "top_p": 0.4}
    assert engine.calls[0]["prompt"] == [1, 2, 3]
    assert engine.closed


def test_eos_and_length_keep_actual_engine_counts(engine):
    from soup_cli.utils.mlx_generation import stream_native

    engine.responses = [_response("one", 1), _response("", 2, "stop")]
    chunks = list(stream_native(None, engine.tokenizer, [], max_tokens=9, temperature=0))
    assert chunks[-1].completion_tokens == 2 and chunks[-1].finish_reason == "stop"
    assert "".join(c.text for c in chunks) == "one"
    assert engine.sampler["top_p"] == 1.0
    engine.responses = [_response("at-limit", 2, "length")]
    assert (
        list(stream_native(None, engine.tokenizer, [], max_tokens=2))[-1].finish_reason == "length"
    )


def test_closing_stream_closes_native_engine(engine):
    from soup_cli.utils.mlx_generation import stream_native

    engine.responses = [_response("first", 1), _response("never", 2, "length")]
    stream = stream_native(None, engine.tokenizer, [], max_tokens=2)
    assert next(stream).text == "first"
    stream.close()
    assert engine.closed


@pytest.mark.parametrize(
    "options,diagnostic",
    [
        ({"top_p": float("nan")}, "top_p"),
        ({"top_p": 2}, "top_p"),
        ({"stop": ("",)}, "stop"),
        ({"stop": ("a",) * 5}, "stop"),
        ({"max_prompt_tokens": 2}, "input token limit"),
    ],
)
def test_bad_generation_options_do_not_call_engine(engine, options, diagnostic):
    from soup_cli.utils.mlx_generation import stream_native

    with pytest.raises(ValueError, match=diagnostic):
        list(stream_native(None, engine.tokenizer, [], **options))
    assert not engine.calls


def test_nonempty_prompt_and_legacy_terminal_count_contract(engine):
    from soup_cli.utils.mlx_generation import stream_native

    engine.tokenizer.encode = lambda *a, **k: []
    with pytest.raises(ValueError, match="nonempty"):
        list(stream_native(None, engine.tokenizer, []))
    engine.tokenizer.encode = lambda *a, **k: [1]
    engine.responses = [SimpleNamespace(text="old", generation_tokens=1)]
    result = list(stream_native(None, engine.tokenizer, [], max_tokens=1))
    assert result[-1].finish_reason == "length" and result[-1].completion_tokens == 1


def test_zero_top_p_is_refused_instead_of_disabling_native_nucleus_filtering(engine):
    from soup_cli.utils.mlx_generation import stream_native

    with pytest.raises(ValueError, match="zero disables"):
        list(stream_native(None, engine.tokenizer, [], top_p=0))
    assert not engine.calls


def test_the_request_seed_reaches_the_native_rng(engine, monkeypatch):
    from soup_cli.utils.mlx_generation import stream_native

    seeded = []
    core = ModuleType("mlx.core")
    core.random = SimpleNamespace(seed=seeded.append)
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    engine.responses = [_response("x", 1, "length")]
    list(stream_native(None, engine.tokenizer, [], max_tokens=1, seed=1234))
    assert seeded == [1234]


@pytest.mark.parametrize(
    "encoded,options,diagnostic",
    [([], {}, "nonempty"), ([1, 2, 3], {"max_prompt_tokens": 2}, "input token limit")],
)
def test_prompt_refusals_raise_the_native_prompt_error(engine, encoded, options, diagnostic):
    from soup_cli.utils.mlx_generation import NativePromptError, stream_native

    engine.tokenizer.encode = lambda *args, **kwargs: encoded
    with pytest.raises(NativePromptError, match=diagnostic):
        list(stream_native(None, engine.tokenizer, [], **options))
    assert not engine.calls
