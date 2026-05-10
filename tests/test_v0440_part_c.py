"""v0.44.0 Part C — UI plugin registry + env knob tests."""

from __future__ import annotations

import pytest

from soup_cli.ui.plugins import (
    clear_tabs,
    get_tab,
    list_tabs,
    register_tab,
)
from soup_cli.utils.ui_env import UiEnv, resolve_ui_env


@pytest.fixture(autouse=True)
def _clean_tabs():
    """Each test starts with an empty tab registry."""
    clear_tabs()
    yield
    clear_tabs()


# --- plugin registry --------------------------------------------------------

def test_register_tab_happy():
    spec = register_tab(name="my-tab", title="My Tab", render=lambda: "hi")
    assert spec.name == "my-tab"
    assert spec.title == "My Tab"
    assert get_tab("my-tab") is spec


def test_register_tab_immutable_view():
    register_tab(name="t1", title="T1", render=lambda: "x")
    snapshot = list_tabs()
    with pytest.raises(TypeError):
        snapshot["x"] = None  # type: ignore[index]


def test_register_tab_rejects_invalid_name():
    with pytest.raises(ValueError):
        register_tab(name="Bad Name", title="x", render=lambda: "y")
    with pytest.raises(ValueError):
        register_tab(name="-leading", title="x", render=lambda: "y")
    with pytest.raises(ValueError):
        register_tab(name="x" * 99, title="x", render=lambda: "y")


def test_register_tab_rejects_invalid_title():
    with pytest.raises(ValueError):
        register_tab(name="t", title="", render=lambda: "x")
    with pytest.raises(ValueError):
        register_tab(name="t", title="bad\x00", render=lambda: "x")
    with pytest.raises(ValueError):
        register_tab(name="t", title="x" * 200, render=lambda: "x")


def test_register_tab_rejects_non_callable_render():
    with pytest.raises(TypeError):
        register_tab(name="t", title="x", render="not callable")  # type: ignore[arg-type]


def test_register_tab_idempotent_for_same_spec():
    fn = lambda: "y"  # noqa: E731
    spec1 = register_tab(name="t", title="T", render=fn)
    spec2 = register_tab(name="t", title="T", render=fn)
    assert spec1 == spec2


def test_register_tab_rejects_re_register_with_different_spec():
    register_tab(name="t", title="T", render=lambda: "x")
    with pytest.raises(ValueError, match="already registered"):
        register_tab(name="t", title="OTHER", render=lambda: "y")


def test_register_tab_too_many():
    for idx in range(32):
        register_tab(name=f"t{idx}", title=f"T{idx}", render=lambda: "x")
    with pytest.raises(RuntimeError, match="too many tabs"):
        register_tab(name="overflow", title="x", render=lambda: "x")


def test_get_tab_unknown_returns_none():
    assert get_tab("nope") is None
    assert get_tab(123) is None  # type: ignore[arg-type]


# --- UI env knobs -----------------------------------------------------------

def test_resolve_ui_env_empty():
    env = resolve_ui_env({})
    assert env == UiEnv(None, None, None, None, None)


def test_resolve_ui_env_full():
    env = resolve_ui_env(
        {
            "API_HOST": "127.0.0.1",
            "API_PORT": "8080",
            "API_KEY": "secret-key-1234",
            "GRADIO_HOST": "0.0.0.0",
            "GRADIO_PORT": "7860",
        }
    )
    assert env.api_host == "127.0.0.1"
    assert env.api_port == 8080
    assert env.api_key == "secret-key-1234"
    assert env.gradio_host == "0.0.0.0"
    assert env.gradio_port == 7860


def test_resolve_ui_env_invalid_port():
    with pytest.raises(ValueError):
        resolve_ui_env({"API_PORT": "0"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_PORT": "99999"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_PORT": "not-int"})


def test_resolve_ui_env_invalid_host():
    with pytest.raises(ValueError):
        resolve_ui_env({"API_HOST": "bad host with spaces"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_HOST": "x\x00bad"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_HOST": "x" * 300})


def test_resolve_ui_env_blank_treated_as_missing():
    env = resolve_ui_env({"API_HOST": "  ", "API_KEY": "  "})
    assert env.api_host is None
    assert env.api_key is None


def test_resolve_ui_env_invalid_key():
    with pytest.raises(ValueError):
        resolve_ui_env({"API_KEY": "k\x00ey"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_KEY": "k" * 1000})


def test_ui_env_frozen():
    env = UiEnv(None, None, None, None, None)
    with pytest.raises(Exception):
        env.api_host = "x"  # type: ignore[misc]
