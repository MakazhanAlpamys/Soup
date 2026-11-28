"""Version-independent tests for MiniLLM's KV-cache capability probe (#3381, #3385).

Newer Transformers forwards (5.24.1 included) declare ``cache`` rather than
``past_key_values``; the deprecation decorator for the old spelling names
5.40.0. These tests use real forward signatures for both spellings, for a
positional-only cache, a ``**cache`` catch-all, and a model that only swallows
``**kwargs``, so the cached path is chosen by the declared signature rather
than by whichever Transformers is installed.
"""

from __future__ import annotations

from types import SimpleNamespace

from soup_cli.utils.minillm import (
    _kv_cache_parameter,
    _output_cache,
    _supports_kv_cache,
)


def legacy_forward(input_ids=None, past_key_values=None, use_cache=None, **kwargs):
    return None


def renamed_forward(input_ids=None, cache=None, use_cache=None, **kwargs):
    return None


def positional_cache_forward(input_ids=None, cache=None, /, use_cache=None, **kwargs):
    return None


def variadic_cache_forward(input_ids=None, use_cache=None, **cache):
    return None


def kwargs_only_forward(input_ids=None, use_cache=None, **kwargs):
    return None


def renamed_without_use_cache(input_ids=None, cache=None, **kwargs):
    return None


def _model(forward) -> object:
    return SimpleNamespace(forward=forward)


def test_legacy_parameter_is_recognised():
    model = _model(legacy_forward)
    assert _kv_cache_parameter(model) == "past_key_values"
    assert _supports_kv_cache(model) is True


def test_renamed_cache_parameter_is_recognised():
    model = _model(renamed_forward)
    assert _kv_cache_parameter(model) == "cache"
    assert _supports_kv_cache(model) is True


def test_positional_only_cache_is_not_passable_by_keyword():
    model = _model(positional_cache_forward)
    assert _kv_cache_parameter(model) is None
    assert _supports_kv_cache(model) is False


def test_variadic_cache_is_not_a_declared_parameter():
    model = _model(variadic_cache_forward)
    assert _kv_cache_parameter(model) is None
    assert _supports_kv_cache(model) is False


def test_kwargs_only_is_not_a_cache_capability():
    model = _model(kwargs_only_forward)
    assert _kv_cache_parameter(model) is None
    assert _supports_kv_cache(model) is False


def test_a_cache_parameter_without_use_cache_is_not_enough():
    model = _model(renamed_without_use_cache)
    assert _kv_cache_parameter(model) == "cache"
    assert _supports_kv_cache(model) is False


def test_the_output_is_read_under_either_spelling():
    legacy = SimpleNamespace(past_key_values="past")
    named = SimpleNamespace(cache="cache")
    assert _output_cache(legacy, "past_key_values") == "past"
    assert _output_cache(named, "cache") == "cache"
    # Read the other spelling when the declared name is absent.
    assert _output_cache(named, "past_key_values") == "cache"
    assert _output_cache(legacy, "cache") == "past"
    assert _output_cache(SimpleNamespace(), "cache") is None
