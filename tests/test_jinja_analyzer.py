"""Tests for v0.37.0 Part D — JinjaTemplateAnalyzer.

Walks chat-template ASTs to discover which ``message[...]`` fields the
template touches. Used to make ``train_on_responses_only`` masking aware of
non-standard fields (e.g. ``tool_calls``, ``name``, ``weight``).
"""

from __future__ import annotations

import pytest

from soup_cli.utils.jinja_analyzer import (
    DEFAULT_MESSAGE_FIELDS,
    JinjaTemplateAnalyzer,
    extract_message_fields,
)

# ---- extract_message_fields ----------------------------------------------


def test_extracts_role_and_content():
    template = "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
    fields = extract_message_fields(template)
    assert "role" in fields
    assert "content" in fields


def test_extracts_tool_calls_field():
    template = (
        "{% for m in messages %}"
        "{{ m.role }}: {{ m.content }}"
        "{% if m.tool_calls %}{{ m.tool_calls }}{% endif %}"
        "{% endfor %}"
    )
    fields = extract_message_fields(template)
    assert "tool_calls" in fields


def test_extracts_subscript_access():
    # message["content"] form — used by some HF templates
    template = (
        "{% for m in messages %}"
        '{{ m["role"] }}: {{ m["content"] }}'
        "{% endfor %}"
    )
    fields = extract_message_fields(template)
    assert "role" in fields
    assert "content" in fields


def test_extracts_weight_field():
    # message.weight — used by Axolotl per-message training masks
    template = (
        "{% for m in messages %}"
        "{% if m.weight > 0 %}{{ m.content }}{% endif %}"
        "{% endfor %}"
    )
    fields = extract_message_fields(template)
    assert "weight" in fields


def test_returns_empty_set_for_no_message_loop():
    template = "static text with no message loop"
    fields = extract_message_fields(template)
    assert fields == set()


def test_handles_train_field_axolotl_style():
    template = (
        "{% for m in messages %}"
        "{% if m.train %}{{ m.content }}{% endif %}"
        "{% endfor %}"
    )
    fields = extract_message_fields(template)
    assert "train" in fields


def test_rejects_empty_template():
    with pytest.raises(ValueError, match="empty"):
        extract_message_fields("")


def test_rejects_non_string():
    with pytest.raises(TypeError, match="must be str"):
        extract_message_fields(123)  # type: ignore[arg-type]


def test_rejects_null_byte():
    with pytest.raises(ValueError, match="null"):
        extract_message_fields("{{ m.content\x00 }}")


def test_oversize_template_rejected():
    huge = "x" * 200_000
    with pytest.raises(ValueError, match="too large"):
        extract_message_fields(huge)


def test_invalid_jinja_raises():
    template = "{% for m in messages %}{{ m.role"  # unterminated
    with pytest.raises(ValueError, match="parse"):
        extract_message_fields(template)


# ---- JinjaTemplateAnalyzer class -----------------------------------------


def test_analyzer_construct_and_query():
    template = "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
    analyzer = JinjaTemplateAnalyzer(template)
    assert analyzer.has_field("role")
    assert analyzer.has_field("content")
    assert not analyzer.has_field("tool_calls")


def test_analyzer_unknown_field_returns_false():
    template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    analyzer = JinjaTemplateAnalyzer(template)
    assert analyzer.has_field("nonexistent_xyz") is False


def test_analyzer_message_fields_property():
    template = (
        "{% for m in messages %}"
        "{{ m.role }}: {{ m.content }}"
        "{% if m.tool_calls %}T{% endif %}"
        "{% endfor %}"
    )
    analyzer = JinjaTemplateAnalyzer(template)
    fields = analyzer.message_fields
    assert "role" in fields
    assert "content" in fields
    assert "tool_calls" in fields
    # Returned set should be a copy (defence against mutation)
    fields.add("tampered")
    assert "tampered" not in analyzer.message_fields


def test_analyzer_uses_non_standard_fields_helper():
    # Standard fields = role / content. Anything else is "non-standard".
    template = (
        "{% for m in messages %}"
        "{{ m.role }}: {{ m.content }}"
        "{% if m.weight > 0 %}W{% endif %}"
        "{% endfor %}"
    )
    analyzer = JinjaTemplateAnalyzer(template)
    non_standard = analyzer.non_standard_fields()
    assert "weight" in non_standard
    assert "role" not in non_standard
    assert "content" not in non_standard


def test_analyzer_default_fields_constant():
    assert "role" in DEFAULT_MESSAGE_FIELDS
    assert "content" in DEFAULT_MESSAGE_FIELDS
    # Must be frozen — prevent runtime mutation
    assert isinstance(DEFAULT_MESSAGE_FIELDS, frozenset)
    with pytest.raises(AttributeError):
        DEFAULT_MESSAGE_FIELDS.add("x")  # type: ignore[attr-defined]
