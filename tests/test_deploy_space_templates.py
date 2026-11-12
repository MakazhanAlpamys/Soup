"""Regression tests for the built-in Hugging Face Space app templates."""

import ast

import pytest

from soup_cli.commands.deploy import render_space_template


@pytest.mark.parametrize("template", ["gradio-chat", "streamlit-chat"])
def test_rendered_chat_template_has_valid_message_dicts(template: str) -> None:
    rendered = render_space_template(template, model_repo="user/my-model")

    assert 'MODEL_ID = "user/my-model"' in rendered["app.py"]
    module = ast.parse(rendered["app.py"])
    assert not any(
        isinstance(node, ast.Set)
        and any(isinstance(element, ast.Dict) for element in node.elts)
        for node in ast.walk(module)
    )
