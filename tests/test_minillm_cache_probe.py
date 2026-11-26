"""Version-independent tests for MiniLLM's KV-cache capability probe (#3381).

Transformers 5.40 renamed the explicit forward parameter ``past_key_values``
to ``cache``. These tests use tiny stand-in forwards for both spellings and
for a model that only swallows ``**kwargs``, so the cached path is chosen by
the declared signature rather than by whichever Transformers is installed.
"""

from __future__ import annotations

from types import SimpleNamespace

from soup_cli.utils.minillm import (
    _kv_cache_parameter,
    _output_cache,
    _supports_kv_cache,
)


def _model(parameters: tuple[str, ...]) -> object:
    def forward(**kwargs):
        return None

    forward.__signature__ = __import__("inspect").Signature(
        [__import__("inspect").Parameter(name, __import__("inspect").Parameter.KEYWORD_ONLY)
         for name in parameters]
    )
    return SimpleNamespace(forward=forward)


def test_legacy_parameter_is_recognised():
    model = _model(("input_ids", "past_key_values", "use_cache"))
    assert _kv_cache_parameter(model) == "past_key_values"
    assert _supports_kv_cache(model) is True


def test_renamed_cache_parameter_is_recognised():
    model = _model(("input_ids", "cache", "use_cache"))
    assert _kv_cache_parameter(model) == "cache"
    assert _supports_kv_cache(model) is True


def test_kwargs_only_is_not_a_cache_capability():
    model = _model(("input_ids", "kwargs"))
    assert _kv_cache_parameter(model) is None
    assert _supports_kv_cache(model) is False


def test_a_cache_parameter_without_use_cache_is_not_enough():
    model = _model(("input_ids", "cache"))
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
