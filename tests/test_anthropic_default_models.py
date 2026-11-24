"""Anthropic defaults must name a model Anthropic still serves.

claude-3-haiku-20240307 (retired 2026-04-20) and claude-3-5-haiku-20241022
(retired 2026-02-19): https://platform.claude.com/docs/en/about-claude/model-deprecations
"""

from soup_cli.commands import data as data_cmd
from soup_cli.data.providers import anthropic as anthropic_provider
from soup_cli.utils import data_forge

RETIRED = {"claude-3-haiku-20240307", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"}


def test_generate_provider_default_model_is_served():
    assert isinstance(anthropic_provider.DEFAULT_MODEL, str)
    assert anthropic_provider.DEFAULT_MODEL.strip() == anthropic_provider.DEFAULT_MODEL
    assert anthropic_provider.DEFAULT_MODEL
    assert anthropic_provider.DEFAULT_MODEL not in RETIRED


def test_augment_uses_a_served_anthropic_model_by_default(monkeypatch):
    seen = {}

    def fake_make_judge_provider_fn(provider, *, model, base_url, raise_on_error):
        seen["model"] = model
        return lambda prompt: {"text": ""}

    monkeypatch.setattr(data_forge, "make_judge_provider_fn", fake_make_judge_provider_fn)
    data_cmd._load_augment_provider("anthropic", rpm=60)

    assert seen["model"] not in RETIRED
    # One Anthropic default for both commands, so a later edit cannot change one alone.
    assert seen["model"] == anthropic_provider.DEFAULT_MODEL
